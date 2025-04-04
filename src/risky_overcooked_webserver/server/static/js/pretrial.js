// Persistent network connection that will be used to transmit real-time data
var socket = io();


/* ###################################################
Game layout
--------------------------------    ------------
|                              |
|        Game Window           |
|                              |       Root
|                              |     Container
--------------------------------
|      Button Window ()        |
________________________________    -----------
###########################################################*/
const root = document.getElementById("root-container");
// set fixed width of root
root.style.width = "100%";
root.style.height = "480px";
// set outline

const game_window = document.getElementById("game-window");

const root = document.getElementById("root-container");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");

// Get container elements
const consent = document.getElementById("consent");
const participant_information = new participant_information(root);
// const participant_information = document.getElementById("participant-information");

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


class ParticipantInformation {
    constructor(container) {
        this.container = container;
        this.text = [
            `
              <h3 class="text-center">Consent</h3>
              <p>
                I am a graduate student under the direction of Professor Wenlong Zhang in the Ira Fulton Schools of Engineering at Arizona State University.
                I am conducting a research study to investigate human-AI interaction in risky settings.
                <br> <br>
                I am inviting your participation, which will involve playing a series of six games taking an estimated 45 minutes to complete.
                The game will be played in your web browser and require both a mouse and keyboard to complete.
                You have the right not to answer any question, and to stop participation at any time.
                <br> <br>
                Your participation in this study is voluntary.
                If you choose not to participate or to withdraw from the study at any time, there will be no penalty.
                For your participation, you will be compensated up to $15 through the Prolific portal based on performance.
                In order to participate, you must be 18 or older.
                <br> <br>
                Your participation will benefit research in interactive AI for the purpose of improving its ability to model and coordinate with humans in risky settings.
                Specifically, your responses to surveys will investigate how human trust in virtual agents developes under different assumptions about behavior.
                Also, the actions you take during the game will help validate and improve robotic understanding of humans.
                After the experiment is complete, further explanation about how the virtual partner modeled your actions is available if you wish to learn more about how you are contributing to this field.
                There are no foreseeable risks or discomforts to your participation.
                <br> <br>
                Responses to all survey questions, participation date, and the actions you take during the game will be collected.
                To protect your privacy, all personal identifiers will be removed from your data and stored on a secure server.
                Your data will be confidential and the de-identified data collected as a part of this study will not be shared
                with others outside of the research team approved by this study’s IRB or be used for future research purposes or other uses.
                The results of this may be used in reports, presentations, or publications but your name will not be used.
                <br> <br>
                If you have any questions concerning the research study, please contact the research team at: mosmith3@asu.edu.
                If you have any questions about your rights as a subject/participant in this research, or if you feel you have been placed at risk, you can contact the Chair of the Human Subjects Institutional Review Board, through the ASU Office of Research Integrity and Assurance, at (480) 965-6788. Please let me know if you wish to be part of the study.
              </p>
            `
        ];


}


updateInstruction();

