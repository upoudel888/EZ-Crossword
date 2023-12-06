const ham = document.querySelector(".ham");
const navLinks = document.querySelector(".nav-links")

ham.addEventListener('click',()=>{
    if(ham.classList.contains("open")){
        navLinks.classList.remove("scale-in-hor-right");
        navLinks.classList.add("scale-out-hor-right");
        ham.classList.toggle("open")
    }else{
        ham.classList.toggle("open")
        navLinks.classList.remove("scale-out-hor-right");
        navLinks.classList.add("scale-in-hor-right");
        
    }
})