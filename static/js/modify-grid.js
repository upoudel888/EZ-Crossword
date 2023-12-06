import Crossword from "./crossword.js";
// initializing the class
let cw = new Crossword(true); // true means enable grid editing

// buttons
const proceedButton = document.querySelector(".proceed-button");
const decreaseButton = document.querySelector(".decrease-button");
const increaseButton = document.querySelector(".increase-button");
const rotateLeftButton = document.querySelector(".rot-left-button");
const rotateRightButton = document.querySelector(".rot-right-button");
const eraseButton = document.querySelector(".erase-button");


proceedButton.addEventListener('click',()=>{
    cw.makeSolveRequest();
});

decreaseButton.addEventListener('click',()=>{
    cw.decreaseGrid();
});

increaseButton.addEventListener('click',()=>{
    cw.increaseGrid();
});

rotateLeftButton.addEventListener("click",()=>{
    cw.rotateGrid(false);
});

rotateRightButton.addEventListener("click",()=>{
    cw.rotateGrid(true);
});

eraseButton.addEventListener("click",()=>{
    cw.eraseGrid();
});
