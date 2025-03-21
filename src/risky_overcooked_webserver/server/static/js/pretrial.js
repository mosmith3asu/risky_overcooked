// Persistent network connection that will be used to transmit real-time data
var socket = io();



const root = document.getElementById("root-container");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");

// Get container elements
const consent = document.getElementById("consent");
const participant_information = document.getElementById("participant-information");

// Define pages to cycle through
let currentIndex = 0;
const pages = [consent, participant_information];
let current_page = pages[currentIndex];

function updateInstruction() {
    pages.forEach(page => page.style.display = "none");
    current_page = pages[currentIndex];
    current_page.style.display = "block";

    if (current_page === consent) {
        prevBtn.style.display = "none";
        nextBtn.style.display = "block";
        nextBtn.textContent = "Agree to Participate";
    }
    else {
        prevBtn.style.display = "block";
        nextBtn.style.display = "block";
        prevBtn.textContent = "Back";
        nextBtn.textContent = "Next";
    }
    // if (currentIndex === 0) {
    //     prevBtn.style.display = "none";
    //     nextBtn.style.display = "block";
    //     nextBtn.textContent = "Agree to Participate";
    // } else if (currentIndex === instructions.length - 1) {
    //     prevBtn.style.display = "block";
    //     nextBtn.style.display = "block";
    //     nextBtn.textContent = "Agree to Participate";
    // }
    // prevBtn.disabled = currentIndex === 0;
    // nextBtn.disabled = currentIndex === instructions.length - 1;
}

prevBtn.addEventListener("click", () => {
    if (currentIndex > 0) {
        currentIndex--;
        updateInstruction();
    }
});

nextBtn.addEventListener("click", () => {
    if (currentIndex < instructions.length - 1) {
        currentIndex++;
        updateInstruction();
    }
});

updateInstruction();