from django.shortcuts import render,HttpResponse,redirect
from django.http import JsonResponse
import json
import puz
import requests
import tempfile

def get_JSON_from_puz(puz_file):

    # Create a temporary file because puz.read takes file_name as an arguement
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(puz_file.read())
    p = puz.read(temp_file.name)    
    numbering = p.clue_numbering()

    grid = []
    gridnum = []
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index: cell + col_index + 1] == '.':
                grid.append(".")
                gridnum.append(0)
            else:
                grid.append(row_solution[col_index: col_index + 1])
                gridnum.append(0)
                
    across_clues = []
    across_answer = []
    for clue in numbering.across:
        across_answer.append(''.join(p.solution[clue['cell'] + i] for i in range(clue['len'])))
        numbered_clue = str(clue['num']) + ". " + clue['clue']
        across_clues.append(numbered_clue)
        gridnum[int(clue['cell'])] = clue['num']

    down_clues = []
    down_answer = []
    for clue in numbering.down:
        down_answer.append(''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len'])))
        numbered_clue = str(clue['num']) + ". " + clue['clue']
        down_clues.append(numbered_clue)
        gridnum[int(clue['cell'])] = clue['num']

    # final JSON format
    grid_data = {'size': { 'rows': p.height, 'cols': p.width}, 
                 'clues': {'across': across_clues, 'down': down_clues}, 
                 'grid': grid,
                 'gridnums':gridnum,
                 'answers':{'across':across_answer,'down':down_answer}
                }

    return grid_data

# 2D array[[row1],[row2]] where each element is [cell_number,gold_answer,predicted_answer]
# Predicted answer is set to 0 initially
def get_rows_and_clues(grid_data):
    rows = [] 
    across_clues = []
    down_clues = [] 
    try:
        no_of_rows = grid_data['size']['rows']
        no_of_cols = grid_data['size']['cols']

        for i in range(no_of_rows):
            temp = []
            for j in range(no_of_cols):
                temp.append([grid_data['gridnums'][i * no_of_cols + j], grid_data['grid'][i * no_of_cols + j],0]) # prone to overflow wrrors
            rows.append(temp)

        def separate_num_clues(clue):
            arr = clue.split(".")
            return (arr[0],"".join(arr[1:]))

        across_clues = [separate_num_clues(i) for i in grid_data['clues']['across']] # array of (clue_num,clue)
        down_clues = [separate_num_clues(i) for i in grid_data['clues']['down']]

        return rows,across_clues,down_clues
    except:
        return [],[],[]


def solve(request):
    context = { }
    if ( request.method == "GET"):
        return render(request,"Solver/solver.html")
    
    if( request.method == "POST"):

        crossword_file = request.FILES['crossword_file']

        if crossword_file:
    
            # Note that when user uploads an image, JS is used to parse it into corresponding JSON
            # the JSON contains extra fields in that case i.e. :
                 # "gridExtractionStatus":"Passed","ClueExtractionStatus":"Passed","parsedFromImage":"True"
            
            # Json File Uploaded
            if crossword_file.content_type == 'application/json':

                json_file = request.FILES.get("crossword_file")
                if json_file is None:
                    return redirect('/solver')
                
                grid_data = json.loads(json_file.read())
                # saving the data in session before redirecting
                request.session['json'] = json.dumps(grid_data) 

                return redirect('Verify')
                 
            # Puz file Uploaded
            elif crossword_file.content_type == 'application/octet-stream':
                request.session['user_uploaded_image'] = False
                
                puz_file = request.FILES.get("crossword_file")
                if puz_file is None:
                    return HttpResponse('Unparsable puz file format.')
                
                grid_data = get_JSON_from_puz(puz_file)
                # saving the data in session before redirecting
                request.session['json'] = json.dumps(grid_data)

                return redirect('Verify')
            
            else:
                return HttpResponse('Invalid file format.')
                   



def verify(request):

    context = {}
    
    if(request.method == "GET"):
        grid_data = json.loads(request.session.get('json'))

        # Extracting the grid and clues from the JSON
        grid_rows,across_clues,down_clues = get_rows_and_clues(grid_data)
                    
    
        # if clue extraction failed then don't show across and down clues
        if(grid_data.get("clueExtractionStatus") and grid_data.get("clueExtractionStatus") != "Passed"):
            across_clues = [(1,""),(16,""),(17,""),(18,""),(19,""),(20,""),(21,""),(22,""),(23,""),(24,""),(25,""),(26,""),(27,""),(28,""),(29,"")]
            down_clues = [(1,""),(2,""),(3,""),(4,""),(5,""),(6,""),(7,""),(8,""),(9,""),(10,""),(11,""),(12,""),(13,""),(14,""),(15,"")]
            
        # if grid extraction fails then show a 15 * 15 grid by default
        if(grid_data.get("gridExtractionStatus") and grid_data.get("gridExtractionStatus") != "Passed"):
            nums = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],  # default grid to display
                    [16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    [29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,],
                    ]
            
            rows = []
            for i in range(15):
                temp = []
                for j in range(15):
                    temp.append((nums[i][j],"A",0))
                rows.append(temp)
            grid_rows = rows

        # Array of arrays 1st array element contains cell data for 1st and 1st columnrow as [ {grid_num},{grid-value}] format
        context['grid_rows'] = grid_rows 
        context['across_clues'] = across_clues
        context['down_clues'] = down_clues
        context['json'] = json.dumps(grid_data)
        context['solutions'] = 0

        return render(request,"Solver/verify.html",context=context)


def saveModifiedJson(request):
    if(request.method == "POST"):
        received_json = json.loads(request.body.decode('utf-8'))  # decoding byte data to a string
        request.session['json'] = json.dumps(received_json)
        data = {"status": "Success", "message": "JSON saved successfully"}
        return JsonResponse(data)

def saveSolution(request):
    if( request.method == "POST"):
        received_solution = json.loads(request.body.decode('utf-8'))  # decoding byte data to a string
        request.session['solution'] = received_solution[0]
        request.session['evaluations'] = received_solution[1]

        data = {"status": "Success", "message": "Solutions saved successfully"}
        return JsonResponse(data)


def showSolution(request):
    grid_data = json.loads(request.session.get('json'))

    # Extracting the grid and clues from the JSON
    grid_rows,across_clues,down_clues = get_rows_and_clues(grid_data)


    solutions = request.session.get("solution")
    evaluations = request.session.get('evaluations')


    context = {}
    for i in range(0,len(solutions)):
        for j in range(0,len(grid_rows[i])):
            grid_rows[i][j][2] = solutions[i][j] if solutions[i][j] != '' else " "


    context['grid_rows'] = grid_rows 
    context['across_clues'] = across_clues
    context['down_clues'] = down_clues
    context['evalutions'] = evaluations
    context['user_uploaded_image'] = True if  grid_data.get("parsedFromImage") == "True" else False

    return render(request,"Solver/solutions.html",context=context)