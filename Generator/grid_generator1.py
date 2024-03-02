import random
from math import floor
import json

class CrosswordGenerator():

    def __init__(self, grid_size = 7, crossword_type = 'british', black_factor = 6):
        self.crossword_type = crossword_type
        self.grid_size = grid_size
        self.rows = grid_size
        self.cols = grid_size
        self.grid_json_data = None
        self.no_ran_iters = 600
        self.no_max_iters = 600
        self.black_factor = black_factor

        self.grid_array = [[' '] * self.cols for _ in range(self.rows)]

    def generate_crossword(self):
        if self.crossword_type == 'british':
            self.generate_british_style()
            self.returnJSON()
            return self.grid_json_data
        
        else:
            print("Generate American Style Crosswords")
            self.generate_american_style()
            self.returnJSON()
            return self.grid_json_data

    def check_status_british(self, grid, init_row, init_col, even):
        """
            Check the status of the crossword grid by verifying if every alternate white square
            participates in a valid horizontal and vertical series of at least 3.

            Parameters:
            - grid (list): The crossword grid.

            Returns:
            - bool: True if the grid is valid, False otherwise.
        """
        for row in range(max(init_row - 2, 0), min(init_row+2, len(grid))):
            for column in range(max(init_col-2, 0), min(init_col+2, len(grid[0]))):
                if grid[row][column] == ' ':
                    # Check horizontally
                    c1 = self.check_series_british(grid, row, column, 0, -1) + self.check_series_british(grid, row, column, 0, 1) - 1

                    # Check vertically
                    c2 = self.check_series_british(grid, row, column, -1, 0) + self.check_series_british(grid, row, column, 1, 0) - 1

                    # Check if the white cell is at an odd index in both row and column and has at least 3 in a series
                    if even:
                        remaining = 0
                    else:
                        remaining = 1

                    if (row % 2 == remaining and column % 2 == remaining):
                        if not (c1 >= 3 and c2 >= 3):
                            return False
                    else:
                        if not (c1 >= 3 or c2 >= 3):
                            return False
        return True
    
    def check_series_british(self, grid, start_row, start_column, row_increment, column_increment, min_count = 4):
        """
            Check if a valid series of at least min_count exists in a specific direction
            starting from the given position.

            Parameters:
            - grid (list): The crossword grid.
            - start_row (int): Starting row index.
            - start_column (int): Starting column index.
            - row_increment (int): Row direction (1, 0, or -1).
            - column_increment (int): Column direction (1, 0, or -1).
            - min_count (int): Minimum count required for a valid series.

            Returns:
            - count: number of white tiles in a specific direction
        """
        count = 1
        current_row, current_column = start_row + row_increment, start_column + column_increment
        while 0 <= current_row < len(grid) and 0 <= current_column < len(grid[0]):
            if grid[current_row][current_column] == ' ':
                count += 1
                current_row += row_increment
                current_column += column_increment
            else:
                break

        return count
                 

    def generate_british_style(self):
        black_cells = 0
        if random.random()<0.5:
            predefined_rows = range(1, self.rows, 2)
            even = True
        else:
            predefined_rows = range(0, self.rows, 2)
            even = False

        for row in predefined_rows:  # odd_places
            for col in predefined_rows:
                if random.random()<0.7:
                    self.grid_array[row][col] = '.'
                    self.grid_array[self.rows - 1 - row][self.cols - 1 - col] = '.'
                    black_cells += 1

        target_black_cells = int(0.25 * self.rows * self.cols)
        iterations = 0

        while black_cells < target_black_cells and iterations < 100:
            iterations += 1
            empty_rows = [i for i in range(self.rows) if '.' not in self.grid_array[i]]
            empty_columns = [j for j in range(self.cols) if all(self.grid_array[i][j] == ' ' for i in range(self.rows))]

            if empty_rows or empty_columns:
                if empty_rows:
                    random_row = random.choice(empty_rows)
                    random_column = random.choice(range(self.cols))
                else:
                    random_row = random.choice(range(self.rows))
                    random_column = random.choice(empty_columns)

                self.grid_array[random_row][random_column] = '.'

                if self.check_status_british(self.grid_array, random_row, random_column, even):
                    self.grid_array[self.rows - 1 - random_row][self.cols - 1 - random_column] = '.'
                    black_cells += 1
                else:
                    self.grid_array[random_row][random_column] = ' '
            else:
                white_cells = [(i, j) for i in range(self.rows) for j in range(self.cols) if self.grid_array[i][j] == ' ']
                if white_cells:
                    random_cell = random.choice(white_cells)
                    self.grid_array[random_cell[0]][random_cell[1]] = '.'

                    if self.check_status_british(self.grid_array, random_cell[0], random_cell[1], even):
                        self.grid_array[self.rows - 1 - random_cell[0]][self.cols - 1 - random_cell[1]] = '.'
                        black_cells += 1
                    else:
                        self.grid_array[random_cell[0]][random_cell[1]] = ' '

    def generate_american_style(self):
        no_ran_iterations = self.no_max_iters
        if self.grid_size != 4:
            while no_ran_iterations == self.no_max_iters:
                self.grid_array = [[' '] * self.cols for _ in range(self.rows)]
                no_ran_iterations = self.add_black_squares()
        else:
            no_ran_iterations = self.add_black_squares()


    def add_black_squares(self):
        """
        Add black squares to the crossword grid based on probability calculations.

        Parameters:
        - grid (list): The crossword grid.
        - black_tiles (list): List of indices representing black squares.

        Returns:
        - None
        """
        black_tiles = []
        white_tiles = [x for x in range(len(self.grid_array) * len(self.grid_array[0])) if x not in black_tiles]
        needed_black = floor(len(self.grid_array) * len(self.grid_array[0]) / self.black_factor)

        if len(self.grid_array) > 7:
            if random.random() < 0.7:
                symmetry = 'diagonal'
            elif 0.7 <= random.random() < 0.85:
                symmetry = 'vertical'
            else:
                symmetry = 'horizontal'
        else:
            symmetry = 'diagonal'

        iterations = 0

        while iterations < self.no_max_iters:
            if len(black_tiles) >= needed_black:
                break
            # Check for empty columns and rows
            empty_columns = [col for col in range(len(self.grid_array[0])) if all(row[col] == ' ' for row in self.grid_array)]
            empty_rows = [row for row in range(len(self.grid_array)) if all(cell == ' ' for cell in self.grid_array[row])]

            # print(empty_columns, empty_rows)

            # Ensure that empty rows and columns cannot exist for grids larger than size 7
            if len(self.grid_array) > 7:
                empty_columns = [] if len(empty_columns) == len(self.grid_array[0]) else empty_columns
                empty_rows = [] if len(empty_rows) == len(self.grid_array) else empty_rows

            if empty_columns and random.random() < 0.5:
                col = random.choice(empty_columns)
                row = random.randint(0, len(self.grid_array) - 1)
            elif empty_rows:
                row = random.choice(empty_rows)
                col = random.randint(0, len(self.grid_array[0]) - 1)
            else:
                row, col = divmod(random.choice(white_tiles), len(self.grid_array[0]))

            odds_row = (needed_black - len(black_tiles)) / len(self.grid_array) ** 1.5
            odds_col = (needed_black - len(black_tiles)) / len(self.grid_array[0]) ** 1.5

            # Adjust the probability for corners and other cells based on the crossword size
            if len(self.grid_array) > 7:
                if (row, col) in [(0, 0), (0, len(self.grid_array[0]) - 1), (len(self.grid_array) - 1, 0), (len(self.grid_array) - 1, len(self.grid_array[0]) - 1)]:
                    odds_corner = 0.1  # Decrease the probability for corners
                else:
                    odds_corner = 1.5  # Increase the probability for other cells
            else:
                odds_corner = 1.0  # Default odds for smaller crosswords

            if self.grid_array[row][col] == ' ' and random.random() < max(odds_row, odds_col, odds_corner):
                self.grid_array[row][col] = '.'
                sym_row, sym_col = self.get_symmetric_tiles(symmetry, row, col)

                self.grid_array[sym_row][sym_col] = '.'

                if not self.check_status_american(self.grid_array, row, col):
                    self.grid_array[row][col] = ' '
                    self.grid_array[sym_row][sym_col] = ' '
                else:
                    black_tiles.extend([row * len(self.grid_array[0]) + col, sym_row * len(self.grid_array[0]) + sym_col])
                    white_tiles.remove(row * len(self.grid_array[0]) + col)

            iterations += 1

        return iterations

    def get_symmetric_tiles(self, symmetry, row, col):
        if symmetry == 'diagonal':
            sym_row = self.grid_size - 1 - row
            sym_col = self.grid_size - 1 - col
        elif symmetry == 'horizontal':
            sym_row = self.grid_size - 1 - row
            sym_col = col
        else:
            sym_row = row
            sym_col = self.grid_size - 1 - col
        return sym_row, sym_col

    def check_status_american(self, grid, init_row, init_col):

        for row in range(max(init_row-2, 0), min(init_row+2,len(grid))):
            for column in range(max(init_col-2, 0), min(init_col+2,len(grid))):
                if grid[row][column] == ' ':
                    # Check horizontally
                    c1 = self.check_series_american(grid, row, column, 0, -1) + self.check_series_american(grid, row, column, 0, 1) -1

                    # Check horizontally to the right
                    # c1 |= check_series(grid, row, column, 0, 1)

                    # Check vertically upwards
                    c2 = self.check_series_american(grid, row, column, -1, 0) + self.check_series_american(grid, row, column, 1, 0) -1

                    # Check vertically
                    # c2 |= check_series(grid, row, column, 1, 0)
                    if not (c1>=3 and c2>=3):
                        return False
        return True

    def check_series_american(self, grid, start_row, start_column, row_increment, column_increment, min_count = 3):

        count = 1
        current_row, current_column = start_row + row_increment, start_column + column_increment
        while 0 <= current_row < len(grid) and 0 <= current_column < len(grid[start_row]):
            if grid[current_row][current_column] == ' ':
                count += 1
                current_row += row_increment
                current_column += column_increment
            else:
                break

        return count

    def returnJSON(self):
        grid = []
        grid_nums = []
        across_clue_num = []
        down_clue_num = []
        # if (x,y) is present in these array the cell (x,y) is already accounted as a part of answer of across or down
        in_horizontal = []
        in_vertical = []

        num = 0

        for x in range(0, self.cols ):
            for y in range(0, self.rows):

                # if the cell is black there's no need to number
                if self.grid_array[x][y] == '.':
                    grid_nums.append(0)
                    continue

                # if the cell is part of both horizontal and vertical cell then there's no need to number
                horizontal_presence = (x, y) in in_horizontal
                vertical_presence = (x, y) in in_vertical

                # present in both 1 1
                if horizontal_presence and vertical_presence:
                    grid_nums.append(0)
                    continue

                # present in one i.e 1 0
                if not horizontal_presence and vertical_presence:
                    horizontal_length = 0
                    temp_horizontal_arr = []
                    # iterate in x direction until the end of the grid or until a black box is found
                    while x + horizontal_length < self.rows and  self.grid_array[x + horizontal_length][y] != '.':
                        temp_horizontal_arr.append((x + horizontal_length, y))
                        horizontal_length += 1
                    # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                    if horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        num += 1
                        across_clue_num.append(num)
                        grid_nums.append(num)
                        continue
                    grid_nums.append(0)
                # present in one 1 0
                if not vertical_presence and horizontal_presence:
                    # do the same for vertical
                    vertical_length = 0
                    temp_vertical_arr = []
                    # iterate in y direction until the end of the grid or until a black box is found
                    while y + vertical_length < self.cols  and  self.grid_array[x][y+vertical_length] != '.':
                        temp_vertical_arr.append((x, y+vertical_length))
                        vertical_length += 1
                    # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                    if vertical_length > 1:
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        down_clue_num.append(num)
                        grid_nums.append(num)
                        continue
                    grid_nums.append(0)

                if(not horizontal_presence and not vertical_presence):

                    horizontal_length = 0
                    temp_horizontal_arr = []
                    # iterate in x direction until the end of the grid or until a black box is found
                    while x + horizontal_length < self.rows  and  self.grid_array[x + horizontal_length][y] != '.':
                        temp_horizontal_arr.append((x + horizontal_length, y))
                        horizontal_length += 1
                    # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array

                    # do the same for vertical
                    vertical_length = 0
                    temp_vertical_arr = []
                    # iterate in y direction until the end of the grid or until a black box is found
                    while y + vertical_length < self.cols  and  self.grid_array[x][y+vertical_length] != '.':
                        temp_vertical_arr.append((x, y+vertical_length))
                        vertical_length += 1
                    # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array

                    if horizontal_length > 1 and horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        across_clue_num.append(num)
                        down_clue_num.append(num)
                        grid_nums.append(num)
                    elif vertical_length > 1:
                        in_vertical.extend(temp_vertical_arr)
                        num += 1
                        down_clue_num.append(num)
                        grid_nums.append(num)
                    elif horizontal_length > 1:
                        in_horizontal.extend(temp_horizontal_arr)
                        num += 1
                        across_clue_num.append(num)
                        grid_nums.append(num)
                    else:
                        grid_nums.append(0)


        size = { 
                'rows' : self.rows,
                'cols' : self.cols,
                }

        dict = {
            'size' : size,
            'grid' : sum(self.grid_array, []),
            'gridnums': grid_nums,
            'across_nums': down_clue_num,
            'down_nums' : across_clue_num,
            'clues':{
                'across':[],
                'down':[]
            }
        }

        self.grid_json_data = dict


if(__name__=='__main__'):
    crossword_generator = CrosswordGenerator(grid_size = 7, crossword_type = 'british', black_factor = 3)
    crossword_grid = crossword_generator.generate_crossword()
    print(json.dumps(crossword_grid))