from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from .grid_generator import generate_grid
from .grid_generator1 import CrosswordGenerator
import json
# Create your views here.
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

def saveSolution(request):
    if( request.method == "POST"):
        received_solution = json.loads(request.body.decode('utf-8'))  # decoding byte data to a string
        request.session['json'] = json.dumps(received_solution)
        return JsonResponse(received_solution)
    

def generate(request):
    if(request.method == "GET"):
        return render(request,"Generator/Generator.html")
    if(request.method == "POST"):
        row_value = int(request.POST.get('row'))
        crossword_type = str(request.POST.get('crossword_type'))
        print("Requested Crossword type was",crossword_type)
        # CHANGE HERE
        crossword_generator = CrosswordGenerator(grid_size = row_value, crossword_type = crossword_type, black_factor = 3) # 'american'
        crossword_grid = crossword_generator.generate_crossword()
        request.session['json'] = json.dumps(crossword_grid)
        request.session['crossword_type'] = crossword_type
        request.session['dim'] = row_value
        return redirect("verify_result")

def verifyGeneratedGrid(request):
        if(request.method == "GET"):
            context = {}
            grid_data = json.loads(request.session.get("json"))
            print(grid_data)
            if(grid_data):
                grid_rows,across_clues,down_clues = get_rows_and_clues(grid_data)
                print(across_clues,down_clues)
                
                context['crossword_type'] = request.session.get('crossword_type')
                context['crossword_dimension'] = request.session.get('dim')
                context['grid_rows'] = grid_rows 
                context['across_clues'] = across_clues
                context['down_clues'] = down_clues
                context['across_nums'] = grid_data['across_nums']
                context['down_nums'] = grid_data['down_nums']
                context['json'] = json.dumps(grid_data)
                context['solutions'] = 0
                return render(request,"Generator/Verify.html",context=context)
        else:
            return HttpResponse("The Generator did not generate the puzzle.")
        
def showGeneratedCrossword(request):
    if(request.method == "GET"):
        context = {}
        grid_data = json.loads(request.session.get("json"))
        print(grid_data)
        if(grid_data):
            grid_rows,across_clues,down_clues = get_rows_and_clues(grid_data)
            print(across_clues,down_clues)
            
            context['grid_rows'] = grid_rows 
            context['across_clues'] = across_clues
            context['down_clues'] = down_clues
            context['across_nums'] = grid_data['across_nums']
            context['down_nums'] = grid_data['down_nums']
            context['json'] = json.dumps(grid_data)
            context['solutions'] = 0
            return render(request,"Generator/Result.html",context=context)
        else:
            return HttpResponse("The Generator did not generate the puzzle.")

def download_json(request):
    json_data = json.loads(request.session.get('json'))

    if json_data:
        # Converting JSON data to a string
        json_string = json.dumps(json_data, indent=4)

        # Creating an HTTP response with JSON content type
        response = HttpResponse(json_string, content_type='application/json')

        # Set the content-disposition header to force download
        response['Content-Disposition'] = 'attachment; filename="crossword.json"'
        
        return response
    else:
        # Handle case when 'json' key is not found in session
        return HttpResponse("JSON data not found in session.")