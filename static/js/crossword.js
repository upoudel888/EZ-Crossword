export default class Crossword{

    constructor(enableGridEdit){
        // they are just variables and not DOM elements
        this.dimension = document.querySelectorAll(".grid-row").length;
        this.grid = []          // array of arrays 
        this.grid_nums = []     // array of cell numbers starting from top to bottom left to right
        this.across_nums = []   // array of non zero cell numbers part of across clues
        this.across_length = []
        this.down_nums = []     // array of non zero cell numbers part of down clues
        this.down_length = []

        this.acrossCluesWithNums = {} // the clue and clue number extracted from the dom
        this.downCluesWithNums = {}   // these variables do not change


        this.initialize(enableGridEdit);
    }

    initialize(enableGridEdit) {
        // DOM elements 
        this.parent = document.querySelector(".grid");

        // they change when user makes some grid changes
        this.grid_rows = document.querySelectorAll(".grid-row");      // changes are made when user increases or decreases grid dimensions
        this.cells = document.querySelectorAll(".grid-cell");         // changes are made when user rotates the grid  
        this.cellNums = document.querySelectorAll(".cell-num");       // used while assigning new numbers to the cells
        
        // they are used for extracting clues from the dom
        this.downClues =  document.querySelectorAll(".down-clue");    
        this.acrossClues = document.querySelectorAll(".across-clue");
        this.dimensionInfo = document.querySelector(".dim-info");

        // "" means white cell and "." mean black cell in this.grid
        this.grid = this.getGrid();      // uses this.cells to compute the cell number                  
        this.computeGridNum();           // computes this.grid_nums, this.across_nums and this.down_nums using this.grid
        this.getCluesWithNums();         // computes this.acrossCluesWithNums and this.downCluesWithNums
        if(enableGridEdit){
            this.assignNewClues();
            this.addCellEventListener();     // Adds click event listener to the grid cells
        }   
    }

    

    async highlight(){

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
        const acrossCluesDiv = document.querySelector(".across-clues");
        const downCluesDiv = document.querySelector(".down-clues");
        let wrongEncounterCount = {}
        
        
        for( let i in this.grid_nums){ // looping through keys

            
            if(this.across_nums.includes(this.grid_nums[i])){  
                let currentlyHighlightingClue = null;
                let currentlyHighlightingCellsPos = [];
                
                // highlight clue
                let acrossClueElem = document.querySelector(`.across-clue-${this.grid_nums[i]}`);  
                acrossClueElem.lastElementChild.style.backgroundColor = "var(--secondary-blue)";
                acrossClueElem.firstElementChild.style.backgroundColor = "var(--secondary-blue)";
                currentlyHighlightingClue = acrossClueElem;
                
                // scroll to make the clue visible 
                let scrollPosition = acrossClueElem.offsetTop - acrossCluesDiv.offsetTop - (acrossCluesDiv.clientHeight/2); 
                acrossCluesDiv.scrollTop = scrollPosition;
                
                // highlight cells
                let length = this.across_length[this.across_nums.indexOf(this.grid_nums[i])];
                let rowPos = Math.floor(i / this.dimension);
                let colPos = i % this.dimension;
                for(let j = 0; j < length; j++){
                    let pos = rowPos * this.dimension + colPos + j;
                
                    this.cells[pos].style.backgroundColor = "var(--secondary-blue)";
                    this.cells[pos].lastElementChild.style.visibility = "visible";
                    this.cells[pos].classList.remove("hide-for-answer");
                    
                    if(this.cells[pos].classList.contains("wrong-cell")){
                        if(! wrongEncounterCount[pos]){
                            wrongEncounterCount[pos] = 1;
                        }else{
                            wrongEncounterCount[pos] += 1;
                        }
                    }
                    
                    currentlyHighlightingCellsPos.push(pos);
                }
                
                // reverse the process
                await sleep(300);
                
                currentlyHighlightingClue.firstElementChild.style.backgroundColor = "white";
                currentlyHighlightingClue.lastElementChild.style.backgroundColor = "var(--secondary-blue-light)";
                for( let pos1 of currentlyHighlightingCellsPos){
                    if(this.cells[pos1].classList.contains("wrong-cell")){
                        if(wrongEncounterCount[pos1] > 1){
                            this.cells[pos1].style.backgroundColor = "var(--secondary-pink)";
                        }else{
                            this.cells[pos1].style.backgroundColor = "white";
                        }
                    }else{
                        this.cells[pos1].style.backgroundColor = "white";
                    }
                }
                
                currentlyHighlightingClue.style.cursor = "pointer"; // add event listener
                currentlyHighlightingClue.addEventListener("click",()=>{
                    // remove every other highlight classes
                    let highlighted = document.querySelectorAll(".highlight");
                    highlighted.forEach(elem =>{elem.classList.remove("highlight")});
                    
                    currentlyHighlightingClue.classList.toggle("highlight");
                    currentlyHighlightingCellsPos.forEach(pos2 =>{
                        this.cells[pos2].classList.toggle("highlight");
                    })
                    
                })
            }
            
            
            if(this.down_nums.includes(this.grid_nums[i])){
                let currentlyHighlightingClue = null;
                let currentlyHighlightingCellsPos = [];
                // highlight clue
                let downClueElem = document.querySelector(`.down-clue-${this.grid_nums[i]}`);  
                downClueElem.lastElementChild.style.backgroundColor = "var(--secondary-blue)";
                downClueElem.firstElementChild.style.backgroundColor = "var(--secondary-blue)";
                currentlyHighlightingClue = downClueElem;
            
                // scroll to make the clue visible 
                let scrollPosition = downClueElem.offsetTop - downCluesDiv.offsetTop - (downCluesDiv.clientHeight/2); 
                downCluesDiv.scrollTop = scrollPosition;

                // highlight cells
                let length = this.down_length[this.down_nums.indexOf(this.grid_nums[i])];
                let rowPos = Math.floor(i / this.dimension);
                let colPos = i % this.dimension;
                for(let j = 0; j < length; j++){
                    let pos = (rowPos + j) * this.dimension + colPos;

                    this.cells[pos].style.backgroundColor = "var(--secondary-blue)";
                    this.cells[pos].lastElementChild.style.visibility = "visible";
                    this.cells[pos].classList.remove("hide-for-answer");

                    if(this.cells[pos].classList.contains("wrong-cell")){
                        if(! wrongEncounterCount[pos]){
                            wrongEncounterCount[pos] = 1;
                        }else{
                            wrongEncounterCount[pos] += 1;
                        }
                    }

                    currentlyHighlightingCellsPos.push(pos);
                }

                // reverse the process
                await sleep(300);
                
                currentlyHighlightingClue.firstElementChild.style.backgroundColor = "white";
                currentlyHighlightingClue.lastElementChild.style.backgroundColor = "var(--secondary-blue-light)";
                
                for( let pos1 of currentlyHighlightingCellsPos){
                    if(this.cells[pos1].classList.contains("wrong-cell")){
                        if(wrongEncounterCount[pos1] > 1){
                            this.cells[pos1].style.backgroundColor = "var(--secondary-pink)";
                        }else{
                            this.cells[pos1].style.backgroundColor = "white";
                        }
                    }else{
                        this.cells[pos1].style.backgroundColor = "white";
                    }
                }
                
                currentlyHighlightingClue.style.cursor = "pointer"; // add event listener
                currentlyHighlightingClue.addEventListener("click",()=>{
                    // remove every other highlight classes
                    let highlighted = document.querySelectorAll(".highlight");
                    highlighted.forEach(elem =>{elem.classList.remove("highlight")});
                    currentlyHighlightingClue.classList.toggle("highlight");
                    currentlyHighlightingCellsPos.forEach(pos2 =>{
                        this.cells[pos2].classList.toggle("highlight");
                    })

                })
            }

        }
    }

    // when user clicks button to change the grid
    // changes are made to either this.grid_rows or this.cells to reflect to the dom
    // then querySelection is done to update the remaining variables using reinitialize after update
    reinitializeAfterUpdate(){

        this.cells = document.querySelectorAll(".grid-cell");
        this.cellNums = document.querySelectorAll(".cell-num");
        this.grid_rows = document.querySelectorAll(".grid-row")
        
        this.grid = this.getGrid();
        
        this.computeGridNum();  // compute grid number
        this.assignNewNumbers();
        
        this.getCluesWithNums();
        this.addCellEventListener();
        this.assignNewClues();
    }

    updateDimensionInfo(){
        this.dimensionInfo.innerText = `${this.dimension} X ${this.dimension}`;
    }

    // this function is triggered when user clicks proceed button
    async makeSolveRequest(){

        // Making the JSON data ready
        let gridJSON = {
            "size" : {
                "rows" : this.dimension,
                "cols" : this.dimension
            },
            "gridnums" : this.grid_nums,
        }
        
        let grid = [];
        for(let rows of this.grid){
            for(let elem of rows){
                grid.push(elem);
            }
        }
        
        let acrossClues = [];
        let downClues = [];
        let acrossAnswers = [];
        let downAnswers= [];

        // trying to extract from json-hidden div
        // answers are present only if there's answers field in the parsed JSON
        let jsonDiv = document.querySelector(".json-hidden");
        let jsonObj = JSON.parse(jsonDiv.innerHTML);
        
        for(let clue_num of this.across_nums){
            acrossClues.push(String(clue_num) + ". " + this.acrossCluesWithNums[clue_num]);
            acrossAnswers.push("A".repeat(this.across_length[this.across_nums.indexOf(clue_num)]));
        }
        for(let clue_num of this.down_nums){
            downClues.push(String(clue_num) + ". " + this.downCluesWithNums[clue_num]);
            downAnswers.push("A".repeat(this.down_length[this.down_nums.indexOf(clue_num)]));
        }

        // check if there's answers key in the json
        // yes ? assign answer from json : fill with dummy ones i.e. all AAAAAA's
        if(jsonObj.hasOwnProperty("answers")){
            gridJSON['answers'] = jsonObj["answers"];
        }else{
            gridJSON['answers'] = {
                "across" : acrossAnswers,
                "down": downAnswers
            };
        }
        
        gridJSON["grid"] = grid;
        gridJSON['clues'] = {
            "across" : acrossClues,
            "down": downClues
        };

        const hero = document.querySelector(".hero");   // to show the loading svg
        hero.classList.toggle("overlay"); 

        
        console.log("Sending solve request for");
        console.log(JSON.stringify(jsonObj));

        // to bypass the 10s serverless function timout on VERCEL, the request to huggingface API
        // is made from the frontend
        const response = await fetch("https://ujjwal123-ez-crossword.hf.space/solve", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(jsonObj)
        })
        const jsonResponse = await response.json();
        
        console.log("The solution is");
        console.log(jsonResponse);
        
        // Now to save the solution in the session the solution is sent to the route /solver/save
        // retrieving the csrf token
        function getCookie(name) {
            let cookieValue = null;
            if (document.cookie && document.cookie !== '') {
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    // Does this cookie string begin with the name we want?
                    if (cookie.substring(0, name.length + 1) === (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
        const csrftoken = getCookie('csrftoken');
        // sending the save post request
        const postResponseToServer = await fetch("/solver/save-solution/",{
            method : "POST",
            headers:{
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken

            },
            body: JSON.stringify(jsonResponse)
        })
        
        hero.classList.toggle("overlay");
        
        const windowLocation = postResponseToServer.url
        window.location.href = windowLocation;

    }


    // ****************** grid manipulation ***************************
    

    // div.grid-cell
    //   |-div.cell-num
    //   |-div.cell-data
    createCell(){

        const gridCellDiv = document.createElement('div'); 
        gridCellDiv.classList.add('grid-cell');
      
        const cellNumDiv = document.createElement('div');
        cellNumDiv.classList.add('cell-num');
      
        const cellDataDiv = document.createElement('div');
        cellDataDiv.classList.add('cell-data');
      
        gridCellDiv.appendChild(cellNumDiv);
        gridCellDiv.appendChild(cellDataDiv);
      
        return gridCellDiv;
    }
    
    // div.grid-row
    //   |- div.grid-cell  | div.grid-cell  | div.grid-cell
    createRow(rowArr){
        const gridRowDiv = document.createElement('div');
        gridRowDiv.classList.add("grid-row");

        for (let cell of rowArr){
            gridRowDiv.appendChild(this.createCell());
        }
        return gridRowDiv
    }

    increaseGrid() {
        if (this.dimension < 30){
            // event listeners become semtically incorrect ( so remove them )
            // later update correct ones using this.reinitializeAfterUpdate function
            this.removeCellEventListener(); 

            // Add a empty cell (" ") to each existing row
            for (let i = 0; i < this.dimension; i++) {
                if(this.dimension % 2 != 0){
                    this.grid_rows[i].appendChild(this.createCell()); //updating dom
                }else{
                    this.grid_rows[i].prepend(this.createCell());
                }
            }
            // Add a new row 
            const newRow = Array(this.dimension + 1).fill(" ");
            if(this.dimension % 2 != 0){
                this.parent.appendChild(this.createRow(newRow)); // updating dom
            }else{
                this.parent.prepend(this.createRow(newRow));    
            }
            this.dimension++;
            this.reinitializeAfterUpdate();  
            this.updateDimensionInfo();
        }
    }

    decreaseGrid() {
        if (this.dimension > 5) {
            this.removeCellEventListener();
            if(this.dimension % 2 == 0){
                // Remove the last row 
                this.parent.removeChild(this.parent.lastElementChild); // updating dom
            }else{
                // Remove the first row 
                this.parent.removeChild(this.parent.firstElementChild); // updating dom
            }

            // Remove the last cell from each remaining row
            for (let i = 0; i < this.dimension ; i++) {
                if(this.dimension % 2 == 0){
                    // Remove the last cell
                    this.grid_rows[i].removeChild(this.grid_rows[i].lastElementChild); // updating dom
                }else{
                    // Remove the first cell 
                    this.grid_rows[i].removeChild(this.grid_rows[i].firstElementChild); // updating dom
                }
            }

            this.dimension--;
            this.reinitializeAfterUpdate();
            this.assignNewNumbers();
            this.updateDimensionInfo();
        }
    }

    rotateGrid(clockwise=false){

        let rotatedTempGrid = [];
        for( let i = 0 ; i < this.dimension; i++){
            rotatedTempGrid.push([]);
            for(let j = this.dimension-1; j >= 0 ; j--){
                rotatedTempGrid[i].push(this.cells[ (this.dimension-1-j) * this.dimension + i]);
            }
        }

        if(clockwise){
            rotatedTempGrid.forEach(row => row.reverse());
        }else{
            rotatedTempGrid.reverse();
        }

        for( let i = 0 ; i < this.dimension; i++){
            for(let j = 0; j < this.dimension ; j++){
                this.cells[i*this.dimension+j].replaceWith(rotatedTempGrid[i][j].cloneNode(true)); // replace with new cells having no new event listeners
            }
        }

        this.reinitializeAfterUpdate();

    }

    eraseGrid(){
        // Remove the last cell from each remaining row
        for (let i = 0; i < this.cells.length ; i++) {
            this.cells[i].classList.remove("dead-cell");
        };
        this.removeCellEventListener();
        this.reinitializeAfterUpdate();
    }


    handleCellClick(cell,index){
        // original clicked position
        let click_x = Math.floor(index / this.dimension);
        let click_y = index % this.dimension;

        // mirror position
        let click_x1 = ( this.dimension - 1  ) - click_x;
        let click_y1 = ( this.dimension - 1 ) - click_y;

        // toggling the class and updating this.grid for original position
        cell.classList.toggle("dead-cell");
        if(cell.classList.contains("dead-cell")){
            this.grid[click_x][click_y] = '.';
        }else{
            this.grid[click_x][click_y] = ' ';
        }

        // toggling the class and updating this.grid for mirror position
        if( !( click_x === click_y && click_x === Math.floor(this.dimension / 2))){
            let cell_mirror = this.cells[click_x1 * this.dimension + click_y1];

            if(cell.classList.contains("dead-cell")){                   // toggle if original is dead cell and mirror is not
                if( ! cell_mirror.classList.contains("dead-cell")){
                    cell_mirror.classList.toggle("dead-cell");
                }
            }else{
                if(cell_mirror.classList.contains("dead-cell")){        // toggle if original is not dead cell and the mirror is
                    cell_mirror.classList.toggle("dead-cell");
                }
            }

            if(cell_mirror.classList.contains("dead-cell")){            // updating this.grid
                this.grid[click_x1][click_y1] = '.';
            }else{
                this.grid[click_x1][click_y1] = ' ';
            }
        }
        // computing new grid nums
        this.computeGridNum();
        // updating the dom with new numbers
        this.assignNewNumbers();
        // updating the dom with new clues
        this.assignNewClues();
    }
    

    // if clicked => toogle class => update this.cell => this.grid => update this.grid_nums => update the dom for grid nums
    
    addCellEventListener(){
        this.cells.forEach((cell,index)=>{
            cell.addEventListener('click',()=>{this.handleCellClick(cell,index)});
        });
    }

    removeCellEventListener(){
        this.cells.forEach((cell)=>{
            cell.replaceWith(cell.cloneNode(true));
        });
    }
    
    // change the UI with the updated grid numbers
    assignNewNumbers(){
        for(let idx in this.grid_nums){
            if(this.grid_nums[idx] === 0){
                this.cellNums[idx].innerHTML = " ";
            }else{
                this.cellNums[idx].innerHTML = this.grid_nums[idx];
            }
        }
    }
    
    addClueEditEventListener(){
        this.acrossClues.forEach((elem)=>{
            elem.lastElementChild.addEventListener("input",()=>{
                this.acrossCluesWithNums[elem.firstElementChild.innerText] = elem.lastElementChild.innerText.trim();
            })
        });
        this.downClues.forEach((elem)=>{
            elem.lastElementChild.addEventListener("input",()=>{
                this.downCluesWithNums[elem.firstElementChild.innerText] = elem.lastElementChild.innerText.trim();
            });
        });
    }

    // get across and down clues from the dom
    getCluesWithNums(){
        for(let elem of this.acrossClues){
            this.acrossCluesWithNums[elem.firstElementChild.innerText] = elem.lastElementChild.innerText;
        }

        for(let elem of this.downClues){
            this.downCluesWithNums[elem.firstElementChild.innerText] = elem.lastElementChild.innerText;
        }
    }

    // delete the previous list of across and down clues
    // change the UI with newly computed across and down grid numbers
    // also update their selectors i.e. this.acrossClues and this.downClues
    assignNewClues(){
        const acrossParent = document.querySelector(".across-clues");
        const downParent = document.querySelector(".down-clues");

        this.acrossClues.forEach((elem)=>{  // deleting previous DOM elements
            acrossParent.removeChild(elem);
        })
        this.downClues.forEach((elem)=>{
            downParent.removeChild(elem);
        })

        this.across_nums.forEach((elem)=>{  // adding new DOM elements
            const divElement = document.createElement('div');
            const clueNumElement = document.createElement('div');
            const clueTextElement = document.createElement('div');
            
            const num = elem;
            let clue = this.acrossCluesWithNums[num];
            if(clue === undefined) clue = " ";

            // adding class List
            divElement.classList.add('across-clue');
            divElement.classList.add(`across-clue-${num}`);

            clueNumElement.classList.add('clue-num');
            clueTextElement.classList.add('clue-text');
            clueTextElement.setAttribute('contenteditable', true);
            clueTextElement.textContent = clue;
            clueNumElement.textContent = num;
            
            // appending the elements
            divElement.appendChild(clueNumElement);
            divElement.appendChild(clueTextElement);
            acrossParent.appendChild(divElement);

        });

        

        // adding new down clues
        this.down_nums.forEach((elem)=>{
            const divElement = document.createElement('div');
            const clueNumElement = document.createElement('div');
            const clueTextElement = document.createElement('div');
            
            const num = elem;
            let clue = this.downCluesWithNums[num];
            if(clue === undefined) clue = " ";

            // adding class List
            divElement.classList.add('down-clue');
            divElement.classList.add(`down-clue-${num}`);

            clueNumElement.classList.add('clue-num');
            clueTextElement.classList.add('clue-text');
            clueTextElement.setAttribute('contenteditable', true);
            clueTextElement.textContent = clue;
            clueNumElement.textContent = num;
            
            // appending the elements
            divElement.appendChild(clueNumElement);
            divElement.appendChild(clueTextElement);
            downParent.appendChild(divElement);
        });

        // updating these variables to remove in the next condition
        this.acrossClues = document.querySelectorAll(".across-clue");
        this.downClues =  document.querySelectorAll(".down-clue");

        this.addClueEditEventListener();
    }

    // get's the grid from the DOM once the page is loaded
    // Returns Array of Arrays
    getGrid(){
        // the main arr
        let arr = new Array();

        // row wise array appended into the main arr
        let tempArr = new Array();
        let count = 0;
        for(let cell of this.cells){
            if(cell.classList.contains("dead-cell")){
                tempArr.push(".");
            }else{
                tempArr.push(" ");
            }
            count += 1;
            // checking if a row has been completed
            if((count % this.dimension == 0)){
                arr.push(tempArr)
                tempArr = []
            }
        }
        return arr;
    }

    // computes the grid num from using the grid colors ( it uses this.grid to see the colors )
    // returns an array
    computeGridNum() {
    
        this.grid_nums = [];    // empty class variables
        this.across_nums = [];
        this.across_length = [];
        this.down_length = [];
        this.down_nums = [];

        let in_horizontal = [];
        let in_vertical = [];
        let num = 0;
    
        for (let x = 0; x < this.dimension; x++) {
            for (let y = 0; y < this.dimension; y++) {
                // If there is a black cell, then there's no need to number
                if (this.grid[x][y] === ".") {
                    this.grid_nums.push(0);
                    continue;
                }
    
                // Check if the cell is part of both horizontal and vertical cells
                let horizontal_presence = in_horizontal.some(coord => coord[0] === x && coord[1] === y);
                let vertical_presence = in_vertical.some(coord => coord[0] === x && coord[1] === y);
    
                // If present in both (1 1)
                if (horizontal_presence && vertical_presence) {
                    this.grid_nums.push(0);
                    continue;
                }
    
                // If present in one (1 0)
                if (!horizontal_presence && vertical_presence) {
                    let horizontal_length = 0;
                    let temp_horizontal_arr = [];
    
                    // Iterate in x direction until the end of the grid or until a black box is found
                    while (x + horizontal_length < this.dimension && this.grid[x + horizontal_length][y] !== '.') {
                        temp_horizontal_arr.push([x + horizontal_length, y]);
                        horizontal_length++;
                    }
    
                    // If horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                    if (horizontal_length > 1) {
                        in_horizontal.push(...temp_horizontal_arr);
                        num++;
                        this.across_nums.push(num)
                        this.across_length.push(horizontal_length)
                        this.grid_nums.push(num);
                        continue;
                    }
                    
                    this.grid_nums.push(0);
                }
                
                // If present in one (0 1)
                if (!vertical_presence && horizontal_presence) {
                    let vertical_length = 0;
                    let temp_vertical_arr = [];
                    
                    // Iterate in y direction until the end of the grid or until a black box is found
                    while (y + vertical_length < this.dimension && this.grid[x][y + vertical_length] !== '.') {
                        temp_vertical_arr.push([x, y + vertical_length]);
                        vertical_length++;
                    }
                    
                    // If vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                    if (vertical_length > 1) {
                        in_vertical.push(...temp_vertical_arr);
                        this.down_length.push(vertical_length)
                        num++;
                        this.down_nums.push(num)
                        this.grid_nums.push(num);
                        continue;
                    }
                    
                    this.grid_nums.push(0);
                }
                
                // If not present in both (0 0)
                if (!horizontal_presence && !vertical_presence) {
                    let horizontal_length = 0;
                    let temp_horizontal_arr = [];
                    let vertical_length = 0;
                    let temp_vertical_arr = [];
    
                    // Iterate in x direction until the end of the grid or until a black box is found
                    while (x + horizontal_length < this.dimension && this.grid[x + horizontal_length][y] !== '.') {
                        temp_horizontal_arr.push([x + horizontal_length, y]);
                        horizontal_length++;
                    }
    
                    // Iterate in y direction until the end of the grid or until a black box is found
                    while (y + vertical_length < this.dimension && this.grid[x][y + vertical_length] !== '.') {
                        temp_vertical_arr.push([x, y + vertical_length]);
                        vertical_length++;
                    }
    
                    // If both horizontal and vertical lengths are greater than 1, then update both in_horizontal and in_vertical arrays
                    if (horizontal_length > 1 && vertical_length > 1) {
                        in_horizontal.push(...temp_horizontal_arr);
                        in_vertical.push(...temp_vertical_arr);
                        num++;
                        this.across_nums.push(num);
                        this.across_length.push(horizontal_length);
                        this.down_length.push(vertical_length);
                        this.down_nums.push(num);
                        this.grid_nums.push(num);
                    }
                    // If only vertical length is greater than 1, then update in_vertical array
                    else if (vertical_length > 1) {
                        in_vertical.push(...temp_vertical_arr);                        
                        num++;
                        this.down_length.push(vertical_length);
                        this.down_nums.push(num);
                        this.grid_nums.push(num);
                    }
                    // If only horizontal length is greater than 1, then update in_horizontal array
                    else if (horizontal_length > 1) {
                        in_horizontal.push(...temp_horizontal_arr);
                        num++;
                        this.across_length.push(horizontal_length);
                        this.across_nums.push(num);
                        this.grid_nums.push(num);
                    } else {
                        this.grid_nums.push(0);
                    }
                }
            }
        }

        // I don't know why my logic is opposite XD
        let temp = this.down_nums;
        this.down_nums = JSON.parse(JSON.stringify(this.across_nums));
        this.across_nums = JSON.parse(JSON.stringify(temp));

        temp = this.down_length;
        this.down_length = JSON.parse(JSON.stringify(this.across_length));
        this.across_length = JSON.parse(JSON.stringify(temp));

    }

}
