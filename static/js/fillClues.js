const fillBtn = document.querySelector(".fill-button");

fillBtn.addEventListener("click",async ()=>{
    const overlay = document.querySelector(".overlay"); // to show the loading svg
    overlay.style.display = "flex";

    let startTime = Date.now();
    let elapsedTime = 0;

    const jobStatus = document.querySelector(".job-status");
    const timeElapsedDiv = document.querySelector(".time-elapsed");

    // updating time elapsed every second
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
        }, 1000);

    let jsonDiv = document.querySelector(".json-hidden");
    let gridJSON = JSON.parse(jsonDiv.innerHTML);
    let checkInterval;

    const response = await fetch(
        "https://ujjwal123-crossword-generator.hf.space/fill",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(gridJSON),
        }
      );
    const jsonResponse = await response.json();

    const checkStatus = async () => {
        const response = await fetch(
          `https://ujjwal123-crossword-generator.hf.space/result/${jsonResponse.job_id}`
        );
        const statusResponse = await response.json();
  
        if (statusResponse.status === "completed") {
          console.log("The result is:", statusResponse.result);
          jobStatus.innerHTML = "Completed";
          timeElapsedDiv.innerHTML = "0s";
  
          clearInterval(checkInterval); // stoppping the polling requests
          clearInterval(timerInterval); // stopping the timer
          showReceivedResult(statusResponse.result);
  
        } else if (statusResponse.status === "processing") {
          jobStatus.innerHTML = "Currently Processing";
        } else if (statusResponse.status === "queued") {
          jobStatus.innerHTML = `Enqueued in ${statusResponse.queue_status["index"]} / ${statusResponse.queue_status["length"]}`;
        } else if(statusResponse.status ==="failed"){
          alert("Filling process failed for some reasons");
          clearInterval(checkInterval); // stoppping the polling requests
          clearInterval(timerInterval); // stopping the timer

          const windowLocation = "/generator/";
          window.location.href = windowLocation;

        }
      };
  
      checkStatus();
  
      // after 45 seconds periodically check the status of the request to check if it is completed
      setTimeout(() => {
        checkInterval = setInterval(checkStatus, 5000);
      }, 10000);
})


async function showReceivedResult(result) {
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

    const csrftoken = getCookie("csrftoken");

    // sending the save post request to save the solution
    const postResponseToServer = await fetch("/generator/save-solution/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrftoken,
      },
      body: JSON.stringify(result),
    });

    const windowLocation = "/generator/result/";
    window.location.href = windowLocation;
  }