import Crossword from "./crossword.js";
// initializing the class
let cw = new Crossword(false); // false means disable grid editing
let viewAllBtn = document.querySelector(".reveal-btn");

viewAllBtn.addEventListener("click",()=>{
    cw.revealAll();
    viewAllBtn.classList.toggle("reveal");
    if(viewAllBtn.classList.contains("reveal")){
        viewAllBtn.firstElementChild.innerHTML = `<i class="fa-solid fa-eye-slash"></i>`
        viewAllBtn.lastElementChild.innerHTML = `Hide Answers`
    }else{
        viewAllBtn.firstElementChild.innerHTML = `<i class="fa-solid fa-eye"></i>`
        viewAllBtn.lastElementChild.innerHTML = `Reveal Answers`
    }
})

cw.highlight();

