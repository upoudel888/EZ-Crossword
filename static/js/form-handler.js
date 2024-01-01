// handling drag and drop feature
const dropContainer = document.getElementById("dropcontainer");
const fileInput = document.getElementById("images");

dropContainer.addEventListener(
  "dragover",
  (e) => {
    e.preventDefault();
  },
  false
);

dropContainer.addEventListener("dragenter", () => {
  dropContainer.classList.add("drag-active");
});

dropContainer.addEventListener("dragleave", () => {
  dropContainer.classList.remove("drag-active");
});

dropContainer.addEventListener("drop", (e) => {
  e.preventDefault();
  dropContainer.classList.remove("drag-active");
  fileInput.files = e.dataTransfer.files;
});

// retrieving the CSRF token
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      // Does this cookie string begin with the name we want?
      if (cookie.substring(0, name.length + 1) === name + "=") {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

// handling default form submit behaviour

function handleSubmit(event) {
  event.preventDefault(); // Prevents the default form submission

  const form = event.target;
  const fileInput = form.querySelector("#images");
  const file = fileInput.files[0];
  const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];

  if (allowedTypes.includes(file.type)) {
    const overlay = document.querySelector(".overlay"); // to show the loading svg
    overlay.style.display = "flex";

    let startTime = Date.now();
    let elapsedTime = 0;

    const timeElapsedDiv = document.querySelector(".time-elapsed");
    const jobStatusDiv = document.querySelector(".job-status-contianer");
    jobStatusDiv.style.display = "none";

    const timerInterval = setInterval(() => {
      elapsedTime = Date.now() - startTime;
      const minutes = Math.floor(elapsedTime / 60000);
      const seconds = ((elapsedTime % 60000) / 1000).toFixed(0);
      let timeText = "";
      if (minutes) {
        timeText = `${minutes}m ${seconds}s`;
      } else {
        timeText = `${seconds}s`;
      }
      timeElapsedDiv.innerHTML = timeText;
    }, 1000); // Update time every second

    const formData = new FormData();
    formData.append("file", file);

    console.log("Sending request to digitize crossword");

    fetch("https://ujjwal123-digitizegrid.hf.space/parseImage/", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((jsonData) => {
        console.log("Received JSON:", jsonData);
        const jsonBlob = new Blob([JSON.stringify(jsonData)], {
          type: "application/json",
        });
        formData.delete("file");
        formData.append("crossword_file", jsonBlob, "parsed.json");

        const csrftoken = getCookie("csrftoken");

        fetch(form.getAttribute("action"), {
          method: "POST",
          body: formData,
          headers: {
            "X-CSRFToken": csrftoken,
          },
        })
          .then((response) => {
            // handling the redirect response from django server
            clearInterval(timerInterval); // stopping the timer
            overlay.style.display = "none";

            const windowLocation = response.url;
            window.location.href = windowLocation;
          })
          .catch((error) => {
            clearInterval(timerInterval); // stopping the timer
            overlay.style.display = "none";
            alert("Form submission error:", error);
          });
      })
      .catch((error) => {
        clearInterval(timerInterval); // stopping the timer
        overlay.style.display = "none";
        alert("Error parsing image:", error);
      });
  } else {
    form.submit(); // If it's not an image file, proceed with default form submission
  }
}
