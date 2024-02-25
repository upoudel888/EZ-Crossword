import Crossword from "./crossword.js";
// initializing the class
let cw = new Crossword(false); // false means disable grid editing
let viewAllBtn = document.querySelector(".reveal-btn");
let viewFirstPassBtn = document.querySelector(".checkbox");
let firstPassElements = document.querySelectorAll(".first-pass");
let secondPassElements = document.querySelectorAll(".second-pass");
let correctedCells = document.querySelectorAll(".corrected-cell");

firstPassElements.forEach((elem) => {
  elem.style.display = "none";
});

let firstPassRevealBtnStatus = false;

viewAllBtn.addEventListener("click", () => {
  cw.revealAll();
  viewAllBtn.classList.toggle("reveal");
  if (viewAllBtn.classList.contains("reveal")) {
    viewAllBtn.firstElementChild.innerHTML = `<i class="fa-solid fa-eye-slash"></i>`;
    viewAllBtn.lastElementChild.innerHTML = `Hide Answers`;
  } else {
    viewAllBtn.firstElementChild.innerHTML = `<i class="fa-solid fa-eye"></i>`;
    viewAllBtn.lastElementChild.innerHTML = `Reveal All`;
  }
});

viewFirstPassBtn.addEventListener("click", () => {
  if (firstPassRevealBtnStatus == false) {
    firstPassRevealBtnStatus = true;
    firstPassElements.forEach((elem) => {
      elem.style.display = "block";
    });
    secondPassElements.forEach((elem) => {
      elem.style.display = "none";
    });
    correctedCells.forEach((elem)=>{
        elem.style.backgroundColor = "var(--secondary-pink)";
    })
  } else {
    firstPassRevealBtnStatus = false;
    firstPassElements.forEach((elem) => {
      elem.style.display = "none";
    });
    secondPassElements.forEach((elem) => {
      elem.style.display = "block";
    });

    correctedCells.forEach((elem)=>{
        elem.style.backgroundColor = "lightgreen";
    })
  }
});

cw.highlight();
