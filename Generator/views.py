from django.shortcuts import redirect, render
from .generate_grid import generateGrid
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


def generate(request):
    if(request.method == "GET"):
        return render(request,"Generator/Generator.html")
    if(request.method == "POST"):
        context = {}
        row_value = int(request.POST.get('row'))
        grid_data = generateGrid(row_value)
        print(grid_data)
        grid_rows,across_clues,down_clues = get_rows_and_clues(grid_data)
        context['grid_rows'] = grid_rows 
        context['across_clues'] = across_clues
        context['down_clues'] = down_clues
        context['json'] = json.dumps(grid_data)
        context['solutions'] = 0

        return render(request,"Solver/verify.html",context=context)
