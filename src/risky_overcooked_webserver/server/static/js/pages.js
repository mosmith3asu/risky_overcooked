// Persistent network connection that will be used to transmit real-time data
// Prolific redirect
//


/* ###################################################
################### Game layout ######################

--------------------------------    ------------
|                              |
|        Game Window           |
|        620px X 100%          |       Root
|                              |     Container
--------------------------------
|  Button Window (40px X 100%) |
________________________________    -----------

######################################################*/
var socket = io();
var graphics_config;
var tutorial_config;
let currentIndex = 0;
let pages = [];
let other_pages = {};
let nextBtn = null;
let prevBtn = null;
let buttonContainer = null;
const debug_mode = false; // set to true to show debug page
const compensation_amount = 15;
const condition = 0; // 0: model risk-sensitivity, 1: assume rationality first
const total_games = 10; // total number of games played
const table_row_bg = ['#F5FBFF','white'];


const LAYOUTS = ['risky_coordination_ring','risky_multipath'];
const PSLIPS = [0.4,0.15];
const AI_agents = ['RS-ToM','Rational'];

const prolific_data = parse_prolific_data();

function parse_prolific_data() {
    // get url parameters
    var urlParams = new URLSearchParams(window.location.search);
    return {
        'prolific_id': urlParams.get('PROLIFIC_PID'),
        'study_id': urlParams.get('STUDY_ID'),
        'session_id': urlParams.get('SESSION_ID')
    }
}

document.addEventListener("DOMContentLoaded", function () {
    // const prolific_data = parse_prolific_data();



    // const viewport = document.getElementById("viewport");
    tutorial_config = JSON.parse($('#tutorial_config').text());

    const game_window_height = "800px";
    const button_window_height = "40px";

    // Get the root container
    const rootContainer = document.getElementById("root-container");
    rootContainer.style.width = "100%";
    rootContainer.style.height = "100%";

    // Create game window div
    const gameWindow = document.createElement("div");
    gameWindow.id = "game-window";
    gameWindow.style.width = "100%";
    gameWindow.style.height = game_window_height;
    gameWindow.style.display = "flex";
    gameWindow.style.alignItems = "center";
    gameWindow.style.justifyContent = "center";
    // gameWindow.style.backgroundColor = "#ccc";
    gameWindow.style.padding = "10px"
    if (debug_mode) {
        gameWindow.style.border = "2px solid red";
    }

    // Create button container
    buttonContainer = document.createElement("div");
    buttonContainer.id = "button-window";
    buttonContainer.style.padding = "5px";
    buttonContainer.style.height = button_window_height;
    buttonContainer.style.width = "100%";
    buttonContainer.style.display = "flex";
    buttonContainer.style.justifyContent = "center";
    buttonContainer.style.gap = "10px";
    // Create two buttons
    const button_width = "200px";
    const button_height = "100%";

    // let prevBtn = document.createElement("button");
    prevBtn = document.createElement("button");
    prevBtn.id = "prev-btn";
    prevBtn.innerText = "prev-btn";
    prevBtn.innerText = "Previous";
    prevBtn.style.width = button_width;
    prevBtn.style.height = button_height;

    // let nextBtn = document.createElement("button");
    nextBtn = document.createElement("button");
    nextBtn.id = "next-btn";
    nextBtn.innerText = "Next";
    nextBtn.style.width = button_width;
    nextBtn.style.height = button_height;
    // nextBtn.style.display = "none";

    // Append elements
    buttonContainer.appendChild(prevBtn);
    buttonContainer.appendChild(nextBtn);
    rootContainer.appendChild(gameWindow);
    rootContainer.appendChild(buttonContainer);


    initialize_pages(gameWindow);
    // append Page_Consent(gameWindow) to pages
    // pages.push(new Page_Consent(gameWindow));
    // pages.push(new Page_ParticipantInformation(gameWindow));
    // pages.push(new Page_BackgroundSurvey(gameWindow));
    // pages.push(new Page_Washout(gameWindow));
    // pages.push(new Page_Debug(gameWindow));


    // #################################################################
    // Set up scrolling through pages with buttons #####################
    // #################################################################

    prevBtn.addEventListener("click", () => {
        if (currentIndex > 0) {
            currentIndex--;
            updatePage();
        }
    });
    nextBtn.addEventListener("click", () => {
        if (currentIndex < pages.length - 1) {
            currentIndex++;
            updatePage();
        }
    });

    pages[0].show();


});

function initialize_pages(gameWindow) {
    // Pretrial
    // pages.push(new Page_Consent(gameWindow));
    // pages.push(new Page_ParticipantInformation(gameWindow));
    // pages.push(new Page_BackgroundSurvey(gameWindow));
    // pages.push(new Page_RiskPropensityScale(gameWindow))

    // Instructions
    // pages.push(new Page_Section(gameWindow, "Instructions"));
    // pages.push(new Page_Tutorials(gameWindow, 0));
    // pages.push(new Page_Tutorials(gameWindow, 1));
    // pages.push(new Page_Tutorials(gameWindow, 2));
    // pages.push(new Page_Tutorials(gameWindow, 3));


    pages.push(new Page_Section(gameWindow, "Experiment Start"));

    let game_num = 0
    let multigame = new Page_MultiGamePlay(gameWindow);

    //  Game Loop: Partner 1
    pages.push(multigame.add_game('risky_coordination_ring','RS-ToM')); game_num++;
    pages.push(multigame.add_game('risky_coordination_ring','RS-ToM')); game_num++;
    pages.push(multigame.add_game('risky_coordination_ring','RS-ToM')); game_num++;



    // //  Game Loop: Partner 1
    // pages.push(new Page_GameInstructions(gameWindow, game_num));
    // pages.push(new Page_GamePlay(gameWindow, game_num));
    // pages.push(new Page_TrustSurvey(gameWindow, game_num))
    // game_num++;
    //
    // // Game Loop : Partner 2
    // pages.push(new Page_Washout(gameWindow));
    // pages.push(new Page_GameInstructions(gameWindow, game_num));
    // pages.push(new Page_GamePlay(gameWindow, game_num));
    // pages.push(new Page_TrustSurvey(gameWindow, game_num))
    // game_num++;

    // Debrief
    pages.push(new Page_Section(gameWindow, "Post-Experiment Survey"));
    pages.push(new Page_RelativeTrustSurvey(gameWindow));
    pages.push(new Page_Debrief(gameWindow, condition))


    // Other pages not in main game loop
    pages.push(new Page_Debug(gameWindow));
    other_pages["unable_to_participate"] = new Page_UnableToParticipate(gameWindow);

    current_page = pages[currentIndex];


    // Debrief


}

// #################################################################
// Define Info Pages ###############################################
// #################################################################
class Page_Debug {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";

        parent_container.appendChild(this.container);
        this.text = [
            `
                 <h3 class="text-center">Debugging Page</h3>
                  <p>
                   End of current pages created
                  </p>
            `
        ];
        this.container.innerHTML = this.text;
    }

    show() {
        this.container.style.display = "block";
        prevBtn.style.display = "block";
        nextBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        prevBtn.style.display = "block";
    }
}

class Page_Consent {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";

        parent_container.appendChild(this.container);
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
                   Your data will be confidential and the de-identified data collected as a part of this study will not be shared with others outside of the research team approved by this study’s IRB or be used for future research purposes or other uses.
                   For research purposes, an anonymous numeric code will be assigned to your responses.
                   However, your Prolific worker ID number will be temporarily stored in order to pay you for your time;
                   this data will be deleted as soon as it is reasonably possible.
                    The results of this may be used in reports, presentations, or publications but your name will not be used.
                    The U.S. Department of Defense is providing support for the research.
                    The U.S. Department of Defense personnel responsible for the protection of human subjects will have access to research records.
                    <br> <br>
                    If you have any questions concerning the research study, please contact the research team at: mosmith3@asu.edu.
                    Alternatively, you may contact the principal investigator directly at wenlong.zhang@asu.edu.
                    If you have any questions about your rights as a subject/participant in this research, or if you feel you have been placed at risk, you can contact the Chair of the Human Subjects Institutional Review Board,
                    through the ASU Office of Research Integrity and Assurance, at (480) 965-6788. Please let me know if you wish to be part of the study.

                   </p>
            `
        ];
        this.container.innerHTML = this.text;
    }

    show() {
        this.container.style.display = "block";
        // let nextBtn = document.getElementById("next-btn")
        // let prevBtn = document.getElementById("prev-btn")
        // change nextBtn to say "Agree"
        nextBtn.innerText = "Agree";
        nextBtn.style.display = "block";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
    }
}

class Page_ParticipantInformation {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";
        parent_container.appendChild(this.container);
        this.text = [
            `
              <h3 class="text-center">Study Information</h3>
              <p>
                General Information
                <ul>
                    <li>Your participation is expected to not exceed 45 minutes</li>
                    <li>You will be playing a game with a virtual agent using a keyboard</li>
                    <li>Please ensure you are in a distraction free environment before you begin</li>
                    <li><b>Your compensation will depend on your performance during these games<b></li>
                </ul>

                Before the experiment begins:
                <ul>
                    <li>Basic background information will be collected</li>
                    <li>One short surveys about your attitudes towards risk will be given</li>
                    <li>You will then be instructed on how to play the game</li>
                    <li>You will then play a practice round to familiarize yourself with the controls</li>
                </ul>

                During the Experiment
                <ul>
                    <li>You will play a series of 5 games with two AI partners (10 total games)</li>
                    <li>Each games will last a maximum of 1 minute</li>
                    <li>After each game, you will be given a brief survey</li>
                </ul>

                After the Experiment
                <ul>
                    <li>You take a brief survey comparing the two agents</li>
                    <li>You will be given a brief description of your contribution to this research</li>
                    <li>You will be redirected to back to Prolific to receive your compensation</li>
                </ul>
              </p>
            `
        ];
        this.container.innerHTML = this.text;
    }

    show() {
        this.container.style.display = "block";
        nextBtn.innerText = "Next";
        nextBtn.style.display = "block";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
    }
}

class Page_UnableToParticipate {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.width = "75%"
        this.container.style.display = "none";
        parent_container.appendChild(this.container);
        this.text = [
            `
              <h3 class="text-center">Unable to Participate</h3>
              <p>
                Thank you for your interest in this experiment!
                <br> <br>

                Unfortunately, you are outside of the demographic group approved for this study and you will not be able to participate.
                We appreciate your time and if you have any questions, please reach out to the research team with the contact information below:
                <br> <br>
                Email: mosmith3@asu.edu
                <br> <br>
                Please feel free to exit this page or navigate to another site.
            `
        ];
        this.container.innerHTML = this.text;
    }

    show() {
        this.container.style.display = "block";
        nextBtn.innerText = "Next";
        nextBtn.style.display = "block";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
    }
}

class Page_Washout {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";
        parent_container.appendChild(this.container);

        this.text = [
            `
              <h3 class="text-center">"You Have a New Partner"</h3>

             <p>You have completed five games with the first of two partners.</p>
            <p>You are now interacting with <strong>a new partner</strong> with <strong>different capabilities.</strong></p>
            <p>Do not let any of your judgments about your previous partner affect your evaluation of the next one.</p>
            <p>Your responses to the questions after the following games should be <strong>made independently</strong> from the previous responses.</p>
            <hr>
            <p>Before continuing, please select the answer below that best describes how you should view your second partner:</p>

                <input type="radio" name="partner_view" value="incorrect"> Since we are playing the same game, my new partner will be about the same as my last partner.
                <br>
                <input type="radio" name="partner_view" value="correct"> Since my new partner has different capabilities, I should judge them based only on my interactions in the next five games.

            `
        ];
        this.container.innerHTML = this.text;

        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "washout-submit-btn";
        this.submitBtn.innerText = "Submit";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    submit() {
        // check if "partner_view" was selected
        const has_response = document.querySelector('input[name="partner_view"]:checked');
        if (!has_response) {
            alert("Please select an answer before continuing.");
            return;
        }
        // get value of partner_view
        const washout_val = document.querySelector('input[name="partner_view"]:checked').value;

        if (washout_val === "correct") {
            currentIndex++;
            updatePage();
        } else {
            alert("Please re-read the text carefully and select the correct answer.");
        }

    }

    show() {
        this.container.style.display = "block";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
    }
}

class Page_Section {
    constructor(parent_container, txt) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";
        parent_container.appendChild(this.container);

        this.text = [
            `<h3 class="text-center">${txt}</h3>`
        ];
        this.container.innerHTML = this.text;
    }

    show() {
        this.container.style.display = "block";
        nextBtn.style.display = "block";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
    }
}

class Page_Debrief {
    constructor(parent_container, condition) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";
        parent_container.appendChild(this.container);
        const condition_text = {
            0: ["models your risk-sensitivity", "assumes you are rational"],
            1: ["assumes you are rational", "models your risk-sensitivity"]
        };
        this.text = [
            `
              <h3 class="text-center">The Experiment has Ended</h3>
             Congratulations! <b>You will receive the maximum compensation of $${compensation_amount}</b>.
             Previous remarks on your compensation depending on in-game performance was only used to elicit real-world perceptions of risk and vulnerability.
             We value the time of all participants and therefore fairly reward all who complete this study.
             <br> <br>
            You are free to click the <b>“Complete Study”</b> button below to be redirected back to Prolific and where you receive compensation for your participation within 5 business days.
            If you wish to learn more about this study, additional information is provided below:
            <br> <br>
            <i>Your data will be used to determine the effectiveness of an autonomous agent that models human risk-sensitivity when compared to one that assumes their human partner is rational.
            Your first partner ${condition_text[condition][0]} while your second partner ${condition_text[condition][1]}.
            We observed how your coordination and perceptions about your partner changed depending on which agent you are interacting with.
            Using these observations, we hope to improve an agent’s mental model of humans such that it can better understand, predict, and assist humans in cooperative tasks like search and rescue, manufacturing, and, more generally, joint decision-making scenarios.
            </i>
            <br> <br>
            If you have any questions or would like to withdraw your data for any reason, please contact the research team below:
            <br> <br>
            Email: mosmith3@asu.edu
            `
        ];
        this.container.innerHTML = this.text;

        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "debrief-submit-btn";
        this.submitBtn.innerText = "Complete Study";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    submit() {
        alert('ERROR: ADD REDIRECT TO PROLIFIC HERE');
    }

    show() {
        this.container.style.display = "block";
        nextBtn.innerText = "Next";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
    }
}

// #################################################################
// Define Event Pages ##############################################
// #################################################################

class Page_BackgroundSurvey {
    constructor(parent_container) {
        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.display = "none";
        parent_container.appendChild(this.container);
        this.createForm();

        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "background-survey-submit-btn";
        this.submitBtn.innerText = "Submit";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);


    }

    createForm() {
        const form = document.createElement("form");
        form.id = "surveyForm";

        // Title
        const title = document.createElement("h2");
        title.textContent = "Background Survey";
        form.appendChild(title);

        // Description
        const description = document.createElement("p");
        description.innerHTML = "<strong>To the best of your ability, provide the following information about yourself:</strong>";
        form.appendChild(description);

        // Age Field
        const ageLabel = document.createElement("label");
        ageLabel.textContent = "Age: ";
        const ageInput = document.createElement("input");
        ageInput.type = "text";
        ageInput.name = "age";
        ageInput.id = "ageInput";

        // Allow only numeric values (0-9)
        ageInput.addEventListener('input', function (event) {
            event.target.value = event.target.value.replace(/[^0-9]/g, '');
        });

        form.appendChild(ageLabel);
        form.appendChild(ageInput);
        form.appendChild(document.createElement("br"));
        form.appendChild(document.createElement("br"));

        // Sex Field (Label and options in the same line)
        const sexContainer = document.createElement("div");
        sexContainer.style.display = "flex";
        sexContainer.style.alignItems = "center";

        const sexLabel = document.createElement("label");
        sexLabel.textContent = "Sex: ";
        sexContainer.appendChild(sexLabel);

        const sexOptions = ["Male", "Female"];
        sexOptions.forEach(option => {
            const label = document.createElement("label");
            label.style.marginLeft = "10px";

            const radio = document.createElement("input");
            radio.type = "radio";
            radio.name = "sex";  // All radios share the same name, ensuring only one selection
            radio.classList.add("sexOption");
            radio.value = option;

            label.appendChild(radio);
            label.appendChild(document.createTextNode(option));
            sexContainer.appendChild(label);
        });

        form.appendChild(sexContainer);

        // Append form to container
        this.container.appendChild(form);
    }

    isFormComplete() {
        if (debug_mode) {
            return true;
        }
        ;
        const ageInput = document.getElementById("ageInput").value.trim();
        const sexOptions = document.querySelectorAll(".sexOption");
        // const messageContainer = document.getElementById("messageContainer");

        let sexSelected = false;
        sexOptions.forEach(checkbox => {
            if (checkbox.checked) {
                sexSelected = true;
            }
        });

        if (ageInput === "" || !sexSelected) {
            // messageContainer.textContent = "Please fill in all fields before submitting.";
            // messageContainer.style.color = "red";
            return false;
        } else {
            // messageContainer.textContent = "Form submitted successfully!";
            // messageContainer.style.color = "green";
            return true;
        }
    }

    isExcluded() {
        if (debug_mode) {
            return false;
        }
        ;
        const age = document.getElementById("ageInput").value;
        const too_old = age > 65;
        const too_young = age < 18;
        return too_old || too_young


    }

    submit() {
        // get the value of the age field

        if (!this.isFormComplete()) {
            alert("Please fill out all fields before submitting.");
        } else if (this.isExcluded()) {
            renderOtherPage("unable_to_participate");
            nextBtn.style.display = "none";
            prevBtn.style.display = "none";
        } else {
            currentIndex++;
            updatePage();
        }

        // console.log("Submitting background survey...");
        // this.hide();
        // currentIndex++;
        // updatePage();
    }

    show() {
        this.container.style.display = "block";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
    }
}

class Page_RiskPropensityScale {
    constructor(parent_container) {
        this.header = "Pre-Experiment Survey"
        this.name = "RPS"
        this.row_bg = ['#F5F5F5','white'];
        this.container = document.createElement("div");
        this.container.style.display = "none";
        this.container.style.width = "90%";
        this.container.margin = "0 auto";
        parent_container.appendChild(this.container);


        this.question_cell_width = "30%"
        this.radio_cell_width = "5%";
        this.scale_cell_width = "12%";
        this.fontSize = "14px";
        this.cell_border_style = "none";

        this.instructions_txt = "Please indicate the extent to which you agree or disagree with the following statement" +
            " selecting the option you prefer. Please do not think too long before answering;" +
            " usually your first inclination is also the best one."
        this.questions = [
            ["Safety first.", "Totally Disagree", "Totally Agree"],
            ["I do not take risks with my health.", "Totally Disagree", "Totally Agree"],
            ["I prefer to avoid risks.", "Totally Disagree", "Totally Agree"],
            ["I take risks regularly.", "Totally Disagree", "Totally Agree"],
            ["I really dislike not knowing what is going to happen.", "Totally Disagree", "Totally Agree"],
            ["I usually view risks as a challenge.", "Totally Disagree", "Totally Agree"],
            ["I view myself as a…", "Risk Avoider", "Risk Seeker"],
        ];
        // this.container = document.getElementById(containerId);
        this.render();
        this.add_submit_button()
    }

    render() {
        const header = document.createElement("h3");
        header.innerText = this.header;
        this.container.appendChild(header);


        const table = document.createElement("table");
        // remove line color for cells
        table.style.border = "none";
        table.style.fontSize = this.fontSize;
        table.style.padding = "2px";
        // set a fixed table row heigt


        const instructions = document.createElement("p");
        instructions.innerHTML = this.instructions_txt;
        instructions.style.fontSize = this.fontSize;
        instructions.style.marginBottom = "10px";
        instructions.style.marginTop = "10px";
        instructions.style.display = "flex";
        this.container.appendChild(instructions);


        this.questions.forEach((quest, qIndex) => {
            const q = quest[0];
            const qLeftLabel = quest[1];
            const qRightLabel = quest[2];

            const row = document.createElement("tr");
            row.classList.add("question-row");
            row.style.padding = "2px";
            row.style.backgroundColor = table_row_bg[qIndex % 2];


            const labelCell = document.createElement("td");
            labelCell.innerText = q;
            labelCell.style.width = this.question_cell_width;
            labelCell.style.padding = "2px";
            labelCell.style.border = this.cell_border_style;
            row.appendChild(labelCell);

            const startLabel = document.createElement("td");
            startLabel.innerText = qLeftLabel;
            startLabel.style.width = this.scale_cell_width;
            startLabel.style.textAlign = "center";
            startLabel.style.padding = "2px";
            startLabel.style.border = this.cell_border_style;
            row.appendChild(startLabel);

            for (let i = 1; i <= 9; i++) {
                const cell = document.createElement("td");
                cell.style.padding = "2px";
                cell.style.margin = "0";
                cell.style.textAlign = "center";
                cell.style.verticalAlign = "center";
                cell.style.border = this.cell_border_style;


                cell.style.width = this.radio_cell_width;
                const radio = document.createElement("input");
                radio.type = "radio";
                radio.name = `q${qIndex}`;
                radio.value = i;
                // align radio in cell
                radio.style.margin = "0";
                radio.style.padding = "0";
                radio.style.verticalAlign = "center";
                radio.style.textAlign = "center";
                cell.appendChild(radio);
                row.appendChild(cell);
            }

            const endLabel = document.createElement("td");
            endLabel.innerText = qRightLabel;
            endLabel.style.width = this.scale_cell_width;
            endLabel.style.textAlign = "center";
            endLabel.style.border = this.cell_border_style;
            row.appendChild(endLabel);

            table.appendChild(row);
        });

        this.container.appendChild(table);
    }

    collectResponses() {
        const responses = {"ID": this.name};
        this.questions.forEach((q, i) => {
            const selected = document.querySelector(`input[name="q${i}"]:checked`);
            responses[q[0]] = selected ? selected.value : null;
        });
        return responses;
    }

    isFormComplete() {
        if (debug_mode) {
            return true;
        }
        ;
        // verify that each question is answered
        const allQuestions = this.container.querySelectorAll(".question-row");
        let allAnswered = true;

        allQuestions.forEach((question) => {
            const radios = question.querySelectorAll('input[type="radio"]');
            let answered = false;
            radios.forEach((radio) => {
                if (radio.checked) {
                    answered = true;
                }
            });
            if (!answered) {
                allAnswered = false;
            }
        });

        return allAnswered;
    }

    add_submit_button() {
        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "trust-survey-submit-btn";
        this.submitBtn.innerText = "Submit";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    submit() {
        if (!this.isFormComplete()) {
            alert("Please answer all questions before submitting.");
            return;
        }
        // get the values of the questions
        this.emit_responses(this.collectResponses());
        currentIndex++;
        updatePage();


    }

    emit_responses(responses) {
        console.log(responses);
    }

    show() {
        this.container.style.display = "block";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
    }
}

class Page_TrustSurvey {
    constructor(parent_container, num) {
        this.header = "Post-Game Survey"
        this.name = `Trust${num}`;
        this.container = document.createElement("div");
        this.container.style.display = "none";
        this.container.style.width = "90%";
        this.container.margin = "0 auto";
        this.row_bg = ['#F5F5F5','white']
        parent_container.appendChild(this.container);


        this.question_cell_width = "20%"
        this.radio_cell_width = "7%";
        this.fontSize = "14px";
        this.cell_border_style = "none";

        this.instructions_txt = "Please indicate the extent to which you agree or disagree with the following statement selecting the " +
            "option you prefer. Please do not think too long before answering; usually your first inclination is also the best one.";

        // ];
        this.sections = [
            {
                title: "What % of the time will a virtual partner be…",
                questions: ["Dependable", "Reliable", "Unresponsive", "Predictable"]
            },
            {
                title: "What % of the time will a virtual partner…",
                questions: ["Act consistently", "Take too many risks", "Meet the needs of the task", "Perform as expected", "Play too safe"]
            }
        ];
        this.scale = [" 0% ", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"];
        // this.container = document.getElementById(containerId);
        this.render();
        this.add_submit_button()
    }

    render() {
        const header = document.createElement("h3");
        header.innerText = this.header;
        this.container.appendChild(header);

        const ncol = this.scale.length + 1;
        const table = document.createElement("table");
        // remove line color for cells
        table.style.border = "none";
        table.style.fontSize = this.fontSize;
        table.style.padding = "2px";
        table.style.margin = "0 auto";
        // set a fixed table row heigt

        const instructions = document.createElement("p");
        instructions.innerHTML = this.instructions_txt;
        instructions.style.fontSize = this.fontSize;
        instructions.style.marginBottom = "10px";
        instructions.style.marginTop = "10px";
        instructions.style.display = "flex";
        this.container.appendChild(instructions);

        this.sections.forEach((section, secIndex) => {
            // create a merged row for the section title
            // const empty_row = document.createElement("tr");

            const sectionRow = document.createElement("th");
            sectionRow.colSpan = ncol; // Span across all columns
            sectionRow.textContent = section.title;
            sectionRow.style.fontWeight = "bold";
            sectionRow.style.margin = "0";
            sectionRow.style.marginTop = "10px";
            sectionRow.style.padding = "0px";
            sectionRow.style.paddingTop = "10px";
            sectionRow.style.verticalAlign = "bottom";
            sectionRow.style.border = this.cell_border_style;
            table.appendChild(sectionRow);
            // form.appendChild(sectionTitle);

            // creat question row
            section.questions.forEach((q, qIndex) => {
                const row = document.createElement("tr");
                row.classList.add("question-row");
                row.style.padding = "2px";
                row.style.backgroundColor = table_row_bg[qIndex % 2];

                const labelCell = document.createElement("td");
                labelCell.innerText = q;
                labelCell.style.width = this.question_cell_width;
                labelCell.style.padding = "2px";
                labelCell.style.border = this.cell_border_style;
                row.appendChild(labelCell);

                // Create each radio button
                this.scale.forEach((scaleLabel, i) => {
                    const cell = document.createElement("td");
                    cell.style.padding = "2px";
                    cell.style.margin = "0";
                    cell.style.textAlign = "center";
                    cell.style.verticalAlign = "center";
                    cell.style.border = this.cell_border_style;
                    cell.style.width = this.radio_cell_width;

                    const radio = document.createElement("input");
                    radio.type = "radio";
                    radio.name = `${q}`;
                    radio.value = i;
                    radio.style.margin = "0";
                    radio.style.padding = "0";
                    radio.style.verticalAlign = "center";
                    radio.style.textAlign = "center";

                    const radioLabel = document.createElement("label");
                    radioLabel.textContent = scaleLabel;
                    radioLabel.style.fontSize = "12px";
                    radioLabel.style.margin = "0";
                    radioLabel.style.marginRight = "8px";
                    radioLabel.style.marginLeft = "2px";
                    // radioLabel.style.margin = "0";
                    radioLabel.style.padding = "0";
                    radioLabel.style.verticalAlign = "center";
                    radioLabel.style.textAlign = "left";

                    cell.appendChild(radio);
                    cell.appendChild(radioLabel);
                    // radioGroup.appendChild(radioContainer);
                    //
                    //   cell.appendChild(radio);
                    row.appendChild(cell);


                }); // end radio creation
                table.appendChild(row);
            }); // end question row
        });

        this.container.appendChild(table);
    }

    collectResponses() {
        // get the values of the radio button questions
        const questions = this.container.querySelectorAll(".question-row");
        const responses = {"ID": this.name};
        questions.forEach((question) => {
            const questionName = question.querySelector('input[type="radio"]').name;
            const selectedRadio = question.querySelector('input[type="radio"]:checked');
            if (selectedRadio) {
                responses[questionName] = selectedRadio.value;
            }
        });
        return responses;
    }

    isFormComplete() {
        if (debug_mode) {
            return true;
        }
        ;
        // verify that each question is answered
        const allQuestions = this.container.querySelectorAll(".question-row");
        let allAnswered = true;

        allQuestions.forEach((question) => {
            const radios = question.querySelectorAll('input[type="radio"]');
            let answered = false;
            radios.forEach((radio) => {
                if (radio.checked) {
                    answered = true;
                }
            });
            if (!answered) {
                allAnswered = false;
            }
        });

        return allAnswered;
    }

    add_submit_button() {
        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "trust-survey-submit-btn";
        this.submitBtn.innerText = "Submit";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    submit() {
        if (!this.isFormComplete()) {
            alert("Please answer all questions before submitting.");
            return;
        }
        // get the values of the questions
        this.emit_responses(this.collectResponses());
        currentIndex++;
        updatePage();


    }

    emit_responses(responses) {
        console.log(responses);
    }

    show() {
        this.container.style.display = "block";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
    }
}

class Page_RelativeTrustSurvey {
    constructor(parent_container) {
        this.header = "Pre-Experiment Survey"
        this.name = "RPS"
        this.container = document.createElement("div");
        this.container.style.display = "none";
        this.container.style.width = "90%";
        this.container.margin = "0 auto";
        parent_container.appendChild(this.container);


        this.question_cell_width = "30%"
        this.radio_cell_width = "5%";
        this.scale_cell_width = "12%";
        this.fontSize = "14px";
        this.cell_border_style = "none";

        this.instructions_txt = "Please indicate the extent to which you agree or disagree with the following statement" +
            " selecting the option you prefer. Please do not think too long before answering;" +
            " usually your first inclination is also the best one."

        this.sections = [
            {title: "I believe my first partner partner is ____ than the my second partner.",
                questions: ["More dependable", "More reliable", "More predictable"]
            },
            {title: "I believe my first partner _____ than my second partner…",
                questions: [" Took more risks", "Acted more consistently", "Better met the needs of the task","Better performed as expected","Played more safe"]
            }
        ]

        this.render();
        this.add_submit_button()
    }

    render() {
        const header = document.createElement("h3");
        header.innerText = this.header;
        this.container.appendChild(header);


        const table = document.createElement("table");
        // remove line color for cells
        table.style.border = "none";
        table.style.fontSize = this.fontSize;
        table.style.padding = "2px";
        table.style.margin = "0 auto";
        // set a fixed table row heigt


        const instructions = document.createElement("p");
        instructions.innerHTML = this.instructions_txt;
        instructions.style.fontSize = this.fontSize;
        instructions.style.marginBottom = "10px";
        instructions.style.marginTop = "10px";
        instructions.style.display = "flex";
        this.container.appendChild(instructions);


        const ncol = 7;
        this.sections.forEach((section, secIndex) => {
            const sectionRow = document.createElement("th");
            sectionRow.colSpan = ncol; // Span across all columns
            sectionRow.textContent = section.title;
            sectionRow.style.fontWeight = "bold";
            sectionRow.style.margin = "0";
            sectionRow.style.marginTop = "10px";
            sectionRow.style.padding = "0px";
            sectionRow.style.paddingTop = "10px";
            sectionRow.style.verticalAlign = "bottom";
            sectionRow.style.border = this.cell_border_style;
            table.appendChild(sectionRow);

            const questions = section.questions;
            const qLeftLabel = "Totally Disagree";
            const qRightLabel = "Totally Agree";

            questions.forEach((quest, qIndex) => {
                const q = quest;

                const row = document.createElement("tr");
                row.classList.add("question-row");
                row.style.padding = "2px";
                row.style.backgroundColor = table_row_bg[qIndex % 2];


                const labelCell = document.createElement("td");
                labelCell.innerText = q;
                labelCell.style.width = this.question_cell_width;
                labelCell.style.padding = "2px";
                labelCell.style.border = this.cell_border_style;
                row.appendChild(labelCell);

                const startLabel = document.createElement("td");
                startLabel.innerText = qLeftLabel;
                startLabel.style.width = this.scale_cell_width;
                startLabel.style.textAlign = "center";
                startLabel.style.padding = "2px";
                startLabel.style.border = this.cell_border_style;
                row.appendChild(startLabel);

                for (let i = 1; i <= 9; i++) {
                    const cell = document.createElement("td");
                    cell.style.padding = "2px";
                    cell.style.margin = "0";
                    cell.style.textAlign = "center";
                    cell.style.verticalAlign = "center";
                    cell.style.border = this.cell_border_style;


                    cell.style.width = this.radio_cell_width;
                    const radio = document.createElement("input");
                    radio.type = "radio";
                    radio.name = q;
                    radio.value = i;
                    // align radio in cell
                    radio.style.margin = "0";
                    radio.style.padding = "0";
                    radio.style.verticalAlign = "center";
                    radio.style.textAlign = "center";
                    cell.appendChild(radio);
                    row.appendChild(cell);
                }

                const endLabel = document.createElement("td");
                endLabel.innerText = qRightLabel;
                endLabel.style.width = this.scale_cell_width;
                endLabel.style.textAlign = "center";
                endLabel.style.border = this.cell_border_style;
                row.appendChild(endLabel);

                table.appendChild(row);
            });
        });

        this.container.appendChild(table);
    }

    collectResponses() {
        // get the values of the radio button questions
        const questions = this.container.querySelectorAll(".question-row");
        const responses = {"ID": this.name};
        questions.forEach((question) => {
            const questionName = question.querySelector('input[type="radio"]').name;
            const selectedRadio = question.querySelector('input[type="radio"]:checked');
            if (selectedRadio) {
                responses[questionName] = selectedRadio.value;
            }
        });
        return responses;
    }


    isFormComplete() {
        if (debug_mode) {
            return true;
        }
        ;
        // verify that each question is answered
        const allQuestions = this.container.querySelectorAll(".question-row");
        let allAnswered = true;

        allQuestions.forEach((question) => {
            const radios = question.querySelectorAll('input[type="radio"]');
            let answered = false;
            radios.forEach((radio) => {
                if (radio.checked) {
                    answered = true;
                }
            });
            if (!answered) {
                allAnswered = false;
            }
        });

        return allAnswered;
    }
    add_submit_button() {
        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = "trust-survey-submit-btn";
        this.submitBtn.innerText = "Submit";
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    submit() {
        if (!this.isFormComplete()) {
            alert("Please answer all questions before submitting.");
            return;
        }
        // get the values of the questions
        this.emit_responses(this.collectResponses());
        currentIndex++;
        updatePage();


    }

    emit_responses(responses) {
        console.log(responses);
    }

    show() {
        this.container.style.display = "block";
        this.submitBtn.style.display = "block";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
    }
}

class GamePlayTemplate {
    // This is a template for the game play pages
    constructor(parent_container, num) {
        // Must be initialized in child class
        this.ID = null;
        this.layout = null;
        this.header = null;
        this.player0_name = null; // human agent
        this.player1_name = null; // AI agent
        this.overcooked_id = null; // ID of overcooked div
        this.game_type = null; // type of game (tutorial/overcooked)
        this.data_collection = "off"; // "on/off"



        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.minWidth = '500px';
        this.container.style.minHeight = '500px';
        this.container.style.maxWidth = '800px';
        this.container.style.maxHeight = '800px';
        this.container.style.height = "95%";
        this.container.style.display = "none";
        // this.container.id = null;
        if (debug_mode) {
            this.container.style.border = "2px solid red";
        }
        parent_container.appendChild(this.container);
        this.add_submit_button('Continue')

    }

    verify() {
        if (this.ID === null) {console.error(`Unspecified game param ${this.ID}`)};
        // if (this.num === num) {console.error(`Unspecified game param ${this.num}`)};
        if (this.layout === null) {console.error(`Unspecified game param ${this.layout}`)};
        if (this.header === null) {console.error(`Unspecified game param ${this.header}`)};
        if (this.player0_name === null) {console.error(`Unspecified game param ${this.player0_name}`)}; // human agent
        if (this.player1_name === null) {console.error(`Unspecified game param ${this.player1_name}`)}; // AI agent
        if (this.overcooked_id === null) {console.error(`Unspecified game param ${this.overcooked_id}`)}; // ID of overcooked div
        if (this.game_type === null) {console.error(`Unspecified game param ${this.game_type}`)}; // type of game (tutorial/overcooked)
        if (this.data_collection === null) {console.error(`Unspecified game param ${this.game_type}`)}; // type of game (tutorial/overcooked)

    }

    draw_finished_overlay() {
        // Draw start overlay
        this.finished_overlay = document.createElement("div");
        this.finished_overlay.style.display = "flex";
        this.finished_overlay.style.position = "absolute";
        // get the top of the this.overcooked_id container
        const overcooked = document.getElementById(this.overcooked_id);
        const rect = overcooked.getBoundingClientRect();
        this.finished_overlay.style.top = `${rect.top}px`;
        this.finished_overlay.style.left = `${rect.left}px`;
        this.finished_overlay.style.position = "absolute";
        // overlay.style.zIndex = "9999";
        this.finished_overlay.style.backgroundColor = "rgba(0, 0, 0, 0.5)";
        this.finished_overlay.style.width = `${rect.width}px`;
        this.finished_overlay.style.height = `${rect.height}px`;
        this.finished_overlay.style.display = "flex";
        this.finished_overlay.style.alignItems = "center";
        this.finished_overlay.style.justifyContent = "center";
        this.finished_overlay.style.fontSize = "24px";
        this.finished_overlay.style.color = "white";
        this.finished_overlay.style.fontWeight = "bold";
        this.finished_overlay.style.textAlign = "center";
        this.finished_overlay.innerHTML = "Game Finished";
        this.container.appendChild(overlay);
    }
    game_finished() {
        // alert("ERROR: ADD GAME FINISHED QUERY HERE");
        graphics_end();
        disable_key_listener();
        // this.draw_finished_overlay();
        // socket.emit('leave');
        this.submitBtn.style.display = "block";
    }

    render() {
        const header = document.createElement("h3");
        header.innerText = this.header;
        header.style.textAlign = "center";
        header.style.margin = "0";
        // header.style.marginTop = "20px";
        header.style.width = "100%";
        header.style.height = "10%";
        this.container.appendChild(header);

        const overcooked = document.createElement("div");
        // cetner items horizontally and vertically in div
        overcooked.style.display = "flex";
        overcooked.style.justifyContent = "center";
        overcooked.id = this.overcooked_id ;
        overcooked.style.width = "100%";
        overcooked.style.height = "70%";
        overcooked.style.margin = "0 auto";
        overcooked.style.padding = "0";
        this.container.appendChild(overcooked);

    }

    add_submit_button(txt) {
        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = `${this.ID}-btn`;
        this.submitBtn.innerText = txt;
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    emit_game_query() {
        const params = {
            'layouts': [this.layout],
            phaseTwoScore: tutorial_config['tutorialParams']['phaseTwoScore'],
            playerOne: this.player1_name,
            playerZero: this.player0_name,
            dataCollection: this.data_collection
        }
        const data = {
            "params" : params,
            "game_name" : this.game_type,
            'prolific_id': prolific_data['prolific_id']
        };
        // create (or join if it exists) new game
        socket.emit("join", data);
    }

    submit() {
        this.destroy()
        currentIndex++;
        updatePage();
    };

    show() {
        this.emit_game_query();
        // make items in this.container alighn horizontally and stack vertically
        // this.container.style.display = "block";
        this.container.style.display = "flex";
        this.container.style.flexDirection = "column";
        this.container.style.alignItems = "center";
        this.container.style.justifyContent = "center";

        // this.container.style.display = "flex";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
        this.submitBtn.style.display = "none";
        if (debug_mode) { this.submitBtn.style.display = "block"; }
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
        // this.finished_overlay.style.display = "none";
    }

    destroy() {
        // destroys game to save on memory
        this.container.innerHTML = "";
        socket.emit('leave', {})
        // if (debug_mode) {
        //     socket.emit('leave', {})
        // }
    }
}


class Page_GamePlay extends GamePlayTemplate {
    constructor(parent_container, num) {
        super(parent_container, num);
        // Base Class Params
        this.ID = `perform_${num}`;
        this.layout = LAYOUTS[num];
        this.p_slip = PSLIPS[num];
        this.header = `Game ${num}/${total_games}`
        this.player0_name = 'human'; // human agent
        // this.player1_name = 'StayAI'; // AI agent
        this.player1_name = AI_agents[num]; // AI agent
        this.overcooked_id = `${this.ID}_gameplay`; // ID of overcooked div
        this.game_type = 'overcooked'; // type of game (tutorial/overcooked)
        this.data_collection = "on"; // "on/off"

        this.render();
        this.verify();
    }
}


class Page_MultiGamePlay {
    // This is a template for the game play pages
    constructor(parent_container) {
        // Must be initialized in child class

        this.ID = 'multi_game_container';
        this.curr_game = 0;
        this.layouts = [];
        this.player0_name = []; // human agent
        this.player1_name = []; // AI agent
        this.overcooked_id = `${this.ID}_gameplay`; // ID of overcooked div
        this.game_type = 'overcooked'; // type of game (tutorial/overcooked)
        this.is_created = false;

        // Create Page Container
        this.container = document.createElement("div");
        this.container.style.minWidth = '500px';
        this.container.style.minHeight = '500px';
        this.container.style.maxWidth = '800px';
        this.container.style.maxHeight = '800px';
        this.container.style.height = "95%";
        this.container.style.display = "none";
        if (debug_mode) {
            this.container.style.border = "2px solid red";
        }
        parent_container.appendChild(this.container);
        this.render();
        this.add_submit_button('Continue')

    }
    add_game(layout, AI_type) {
        this.layouts.push(layout);
        this.player0_name.push('human'); // human agent
        this.player1_name.push(AI_type); // AI agent
        return this
    }

    game_finished() {
        // alert("ERROR: ADD GAME FINISHED QUERY HERE");
        graphics_end();
        disable_key_listener();
        ++this.curr_game;

        // this.draw_finished_overlay();
        socket.emit('leave',{});
        this.submitBtn.style.display = "block";
    }

    render() {
        this.header = document.createElement("h3");
        this.header.innerText = 'TBD';
        this.header.style.textAlign = "center";
        this.header.style.margin = "0";
        // header.style.marginTop = "20px";
        this.header.style.width = "100%";
        this.header.style.height = "10%";
        this.container.appendChild(this.header);

        const overcooked = document.createElement("div");
        // cetner items horizontally and vertically in div
        overcooked.style.display = "flex";
        overcooked.style.justifyContent = "center";
        overcooked.id = this.overcooked_id ;
        overcooked.style.width = "100%";
        overcooked.style.height = "70%";
        overcooked.style.margin = "0 auto";
        overcooked.style.padding = "0";
        this.container.appendChild(overcooked);

    }

    add_submit_button(txt) {
        // Add submit button for custom event
        this.submitBtn = document.createElement("button");
        this.submitBtn.id = `${this.ID}-btn`;
        this.submitBtn.innerText = txt;
        this.submitBtn.style.width = nextBtn.style.width;
        this.submitBtn.style.height = nextBtn.style.height;
        this.submitBtn.style.display = "none";
        this.submitBtn.addEventListener("click", () => {
            this.submit();
        });
        buttonContainer.appendChild(this.submitBtn);
    }

    create_game(){
        this.is_created = true;
         const params = {
            'layouts': this.layouts,
            // phaseTwoScore: tutorial_config['tutorialParams']['phaseTwoScore'],
            playerOne: this.player1_name[this.curr_game],
            playerZero: this.player0_name[this.curr_game]
        }
        const data = {
            "params" : params,
            "game_name" : this.game_type,
            'prolific_id': prolific_data['prolific_id']
        };
        // create (or join if it exists) new game
        socket.emit("create", data);
    }
    join_game() {
        socket.emit("client_ready", {});
        // socke
        // // const params = {
        // //     'layouts': [this.layout],
        // //     phaseTwoScore: tutorial_config['tutorialParams']['phaseTwoScore'],
        // //     playerOne: this.player1_name,
        // //     playerZero: this.player0_name
        // // }
        // const data = {
        //     // "params" : params,
        //     // "game_name" : this.game_type,
        //     'prolific_id': prolific_data['prolific_id']
        // };
        // // create (or join if it exists) new game
        // socket.emit("join", data);
    }

    submit() {
        // this.destroy()
        currentIndex++;
        updatePage();
    };

    show() {
        if (!this.is_created){this.create_game();}
        this.join_game();

        // this.emit_game_query();
        // make items in this.container alighn horizontally and stack vertically
        // this.container.style.display = "block";
        this.header.innerText = `Game ${this.curr_game+1}/${this.layouts.length}`;
        this.container.style.display = "flex";
        this.container.style.flexDirection = "column";
        this.container.style.alignItems = "center";
        this.container.style.justifyContent = "center";

        // this.container.style.display = "flex";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
        this.submitBtn.style.display = "none";
        if (debug_mode) { this.submitBtn.style.display = "block"; }
    }

    hide() {
        this.container.style.display = "none";
        this.submitBtn.style.display = "none";
        // this.finished_overlay.style.display = "none";
    }

    destroy() {
        // destroys game to save on memory
        this.container.innerHTML = "";
        if (debug_mode) {
            socket.emit('leave', {})
        }
    }
}

class Page_GameInstructions extends GamePlayTemplate {
    constructor(parent_container, num) {
        super(parent_container, num);
        // Base Class Params
        this.ID = `game_instructions_${num}`;
        this.layout = LAYOUTS[num];
        this.p_slip =  PSLIPS[num];
        this.header = `Game ${num}/${total_games} Instructions`
        this.player0_name = 'StayAI'; // human agent
        this.player1_name = 'StayAI'; // AI agent
        this.overcooked_id = `${this.ID}_gameplay`; // ID of overcooked div
        this.game_type = 'overcooked'; // type of game (tutorial/overcooked)


        // Child Class Params
        this.info_div_width = "50%"
        this.layout_display_div = "50%"
        this.content_padding = "10px";
        this.priming_options = [
            "Take a direct route by carry objects through puddles",
            "Take a longer detour around puddles",
            "Pass objects to partner using counter tops"
        ].sort(() => Math.random() - 0.5); // shuffle priming options

        this.render();
        this.verify();


    }
    render() {
        // render game instructions
        const header = document.createElement("h3");
        header.innerText = this.header;
        header.style.textAlign = "center";
        header.style.margin = "20px";
        this.container.appendChild(header);


        const content_div = document.createElement("div");
        content_div.width = "100%";
        content_div.style.display = "flex";
        this.container.appendChild(content_div);
        content_div.style.margin = "0 auto";


        // create game information text: ##################
        const info_div = document.createElement("div");
        info_div.style.display = "flex";
        info_div.style.flexDirection = "column";
        info_div.style.justifyContent = "center";
        info_div.style.width = this.info_div_width;
        info_div.style.margin = "0 auto";
        info_div.style.padding = this.content_padding;

        const info_pslip = document.createElement("p");
        info_pslip.innerHTML = `You have a ${this.p_slip*100}% chance of slipping in a puddle and losing your held object`;
        info_div.appendChild(info_pslip);

        const priming_text = document.createElement("p");
        priming_text.innerHTML = "Before you begin, please select what you believe to be the best strategy in this game:"
        priming_text.style.fontWeight = "bold";
        info_div.appendChild(priming_text);

        // create priming options radio button
        const table = document.createElement("table");
        table.style.border = "none";

        this.priming_options.forEach((option, i) => {
            const row = document.createElement("tr");
            // make background white
            row.style.backgroundColor = table_row_bg[i % 2];
            row.style.padding = "2px";
            table.appendChild(row)

            const cell = document.createElement("td");
            cell.style.padding = "2px";
            cell.style.margin = "0";
            cell.style.textAlign = "center";
            cell.style.verticalAlign = "center";
            cell.style.border = "none";
            // cell.style.width = this.radio_cell_width;

            const radio = document.createElement("input");
            radio.type = "radio";
            radio.name = `priming_${this.layout}`;
            radio.value = option;
            // align radio in cell
            radio.style.margin = "0";
            radio.style.padding = "0";
            radio.style.verticalAlign = "center";
            radio.style.textAlign = "left";
            cell.appendChild(radio);
            row.appendChild(cell);


            const labelCell = document.createElement("td");
            labelCell.innerText = option;
            // labelCell.style.width = this.question_cell_width;
            labelCell.style.padding = "2px";
            labelCell.style.border = "none";
            // labelCell.style.border = this.cell_border_style;
            row.appendChild(labelCell);

        });
        info_div.appendChild(table);
        content_div.appendChild(info_div);

        // Render image of layout################################
        const layout_div = document.createElement("div");
        layout_div.style.display = "flex";
        layout_div.style.justifyContent = "center";
        layout_div.id = this.overcooked_id;
        layout_div.style.width = this.layout_display_div;
        layout_div.style.height = "300px";
        layout_div.style.padding = 0;//this.content_padding;
        layout_div.style.margin = "0 auto";

        if (debug_mode) {
            layout_div.style.border = "2px solid red";
        }
        content_div.appendChild(layout_div);


    };

    submit() {
        if (!this.isFormComplete()) {
            alert("Please answer all questions before submitting.");
            return;
        }
        // get the values of the questions
        this.emit_responses(this.collectResponses());
        this.destroy()
        currentIndex++;
        updatePage();
    };
    emit_responses(response) {
        // sends priming response to server
        console.log(response);
    }

    isFormComplete() {
        if (debug_mode) {
            return true;
        }
        // verify that a radio option is selected
        const selected = document.querySelector(`input[name="priming_${this.layout}"]:checked`);
        return selected;
    }

    collectResponses() {
        // get the values of the radio button questions
        const responses = {"ID": this.ID};
        if (debug_mode) {
            responses["priming"] = "debug";
        } else {
            const selected = document.querySelector(`input[name="priming_${this.layout}"]:checked`);
            responses["priming"] = selected.value;
        }
        return responses;
    }

    show() {
        this.emit_game_query();
        // make items in this.container alighn horizontally and stack vertically
        // this.container.style.display = "block";
        this.container.style.display = "flex";
        this.container.style.flexDirection = "column";
        this.container.style.alignItems = "center";
        this.container.style.justifyContent = "center";

        // this.container.style.display = "flex";
        nextBtn.style.display = "none";
        prevBtn.style.display = "none";
        this.submitBtn.style.display = "block";
        // if (debug_mode) { this.submitBtn.style.display = "block"; }
    }
}

class Page_Tutorials extends GamePlayTemplate {
    constructor(parent_container, num) {
        super(parent_container, num);
        // Base Class Params
        this.ID = `tutorial-${num}`;
        this.layout = tutorial_config['tutorialParams']['layouts'][num]
        this.p_slip = [0.0,0.3,0.9,0.9][num]
        this.header = [
            'Tutorial 1/4: Controls and Objective',
            'Tutorial 2/4: Puddles',
            'Tutorial 3/4: Taking a Detour',
            'Tutorial 3/4: Passing Objects'
        ][num]
        this.player0_name = "human"; // human agent
        this.player1_name = ["TutorialAI","TutorialAI","TutorialAI"][num]; // AI agent
        this.overcooked_id = `${this.ID}_gameplay`; // ID of overcooked div
        this.game_type = 'tutorial'; // type of game (tutorial/overcooked)


        // Child Class Params
        this.instuctions_txt = this.get_instructions()[num]
        this.instuctions_fontsz = "14px";

        this.render()
        this.verify()

    }
    get_instructions() {
        const instruct = [
        `
        <p>Your goal here is to cook and deliver onion soup in order to earn reward.</p>
        <p>Use the <b>arrow keys</b> to move and <b>spacebar</b> to interact with objects</p>
        <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
        <p>You need to <b>Place 3 onions in pot >> Wait for soup to cook >> Bring a Dish to the pot >> Deliver to service window</b></p>
         <p>You will advance when you have delivered a soup</p>
        `,
        `
        <p>Oh no! Someone spilled water on the floor and created a slippery puddle!</p>
        <p>If you enter a paddle while holding an object, you may slip and fall.</p>
        <p>When you slip, you will <b>lose your held object</b>.</p>
        <p>Each game will have a <b>different chance of slipping</b>. Here, it is a 50% chance.</p>
        <p>Try and cook a soup by navigating through the puddle.</p>
         <p>You will advance when you have delivered a soup</p>
        `,
        `
        <p>One option you have is to simply go around the puddles.</p>
        <p>You must decide which is better:</p>
         <ul>
                    <li>Taking a longer path to avoid the puddle</li>
                    <li>Walking through the puddle and risking a slip</li>
           </ul>
        <p>In this game, there is a ${this.p_slip}% chance of slipping.</p>
         <p>You will advance when you have delivered a soup</p>
        `,
        `
        <p>Alternatively, you can rely on your partner and <b>pass objects over the counter</b>.</p>
        <p>In this game, there is a ${this.p_slip}% chance of slipping.</p>
        <p>Therefore, the only reasonable stratagy is to pass items to your partner.</p>
        <p>Try picking up onions/dishes and putting them on the counter where your partner can reach.</p>
        <p>Your partner will then pass a soup to you for delivery.</p>
        <p>You will advance when you have delivered a soup</p>
        `
    ];
        return instruct



    }

    render() {
        const header = document.createElement("h3");
        header.innerText = this.header;
        header.style.margin = "0";
        header.style.textAlign = "center";
        // header.style.margin = "20px";
        header.style.margin = "0";
        header.style.marginTop = "20px";
        header.style.width = "100%";
        header.style.height = "10%";
        this.container.appendChild(header);

        const instructions = document.createElement("div");
        instructions.id = "instructions";
        instructions.style.width = "100%";
        instructions.style.height = "25%";
        instructions.style.margin = "0";
        instructions.innerHTML = this.instuctions_txt;
        instructions.style.fontSize = this.instuctions_fontsz;
        this.container.appendChild(instructions);

        const overcooked = document.createElement("div");
        overcooked.id = this.overcooked_id;
        overcooked.style.display = "flex";
        overcooked.style.justifyContent = "center";
        overcooked.style.width = "100%";
        overcooked.style.maxHeight = "4500px";
        overcooked.style.height = "60%";
        overcooked.style.margin = "0 auto";
        overcooked.style.padding = "0";
        this.container.appendChild(overcooked);

    }
}
// #################################################################
// Page Event Handlers #############################################
// #################################################################
function renderOtherPage(page_name) {
    // loop through all pages and hide them
    for (let i = 0; i < pages.length; i++) {
        pages[i].hide();
    }
    other_pages[page_name].show();
}

function updatePage() {
    // loop through all pages and hide them
    for (let i = 0; i < pages.length; i++) {
        pages[i].hide();
    }
    current_page = pages[currentIndex];
    current_page.show();
}


/* * * * * * * * * * * * *
 * Socket event handlers *
 * * * * * * * * * * * * */
// on socket connection


socket.on('creation_failed', function(data) {
    // Tell user what went wrong
    let err = data['error']
    $("#overcooked").empty();
    $('#overcooked').append(`<h4>Sorry, tutorial creation code failed with error: ${JSON.stringify(err)}</>`);
    $('#try-again').show();
    $('#try-again').attr("disabled", false);
});

socket.on('start_game', function(data) {
    // alert("Game started");
    curr_tutorial_phase = 0;
    graphics_config = {
        // container_id : "overcooked",
        container_id : current_page.overcooked_id,
        start_info : data.start_info
    };
    // $("#overcooked").empty();
    // $('#game-over').hide();
    // $('#try-again').hide();
    // $('#try-again').attr('disabled', true)
    // $('#hint-wrapper').hide();
    // $('#show-hint').text('Show Hint');
    // $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase}/${tutorial_instructions.length}`);
    // $('#game-title').show();
    // $('#tutorial-instructions').append(tutorial_instructions[curr_tutorial_phase]);
    // $('#instructions-wrapper').show();
    // $('#hint').append(tutorial_hints[curr_tutorial_phase]);
    enable_key_listener();
    graphics_start(graphics_config);
});

socket.on('reset_game', function(data) {
     // alert("Game reset");
    alert("Resetting game deprecated");
    // curr_tutorial_phase++;
    // graphics_end();
    disable_key_listener();
    $("#overcooked").empty();
    current_page.game_finished()

    // // $('#tutorial-instructions').empty();
    // // $('#hint').empty();
    // // $("#tutorial-instructions").append(tutorial_instructions[curr_tutorial_phase]);
    // // $("#hint").append(tutorial_hints[curr_tutorial_phase]);
    // // $('#game-title').text(`Tutorial in Progress, Phase ${curr_tutorial_phase + 1}/${tutorial_instructions.length}`);
    // //
    // // let button_pressed = $('#show-hint').text() === 'Hide Hint';
    // // if (button_pressed) {
    // //     $('#show-hint').click();
    // // }
    // graphics_config = {
    //     container_id : "overcooked",
    //     start_info : data.state
    // };
    // graphics_start(graphics_config);
    // enable_key_listener();
});

socket.on('state_pong', function(data) {
    // Draw state update
    drawState(data['state']);
});

socket.on('end_game', function(data) {
    // alert("Game ended");
    current_page.game_finished()
    if (data.status === 'inactive') {
        // Game ended unexpectedly
        $('#error-exit').show();
        // Propogate game stats to parent window
        window.top.postMessage({ name : "error" }, "*");
    } else {
        // Propogate game stats to parent window
        window.top.postMessage({ name : "tutorial-done" }, "*");
    }

});


/* * * * * * * * * * * * * *
 * Game Key Event Listener *
 * * * * * * * * * * * * * */

function enable_key_listener() {
    $(document).on('keydown', function(e) {
        let action = 'STAY'
        switch (e.which) {
            case 37: // left
                action = 'LEFT';
                break;

            case 38: // up
                action = 'UP';
                break;

            case 39: // right
                action = 'RIGHT';
                break;

            case 40: // down
                action = 'DOWN';
                break;

            case 32: //space
                action = 'SPACE';
                break;

            default: // exit this handler for other keys
                return;
        }
        e.preventDefault();
        socket.emit('action', { 'action' : action });
    });
};

function disable_key_listener() {
    $(document).off('keydown');
};

/* * * * * * * * * * * *
 * Game Initialization *
 * * * * * * * * * * * */

// socket.on("connect", function() {
//     // Config for this specific game
//     let data = {
//         "params" : config['tutorialParams'],
//         "game_name" : "tutorial"
//     };
//
//     // create (or join if it exists) new game
//     socket.emit("join", data);
// });


/* * * * * * * * * * *
 * Utility Functions *
 * * * * * * * * * * */

var arrToJSON = function(arr) {
    let retval = {}
    for (let i = 0; i < arr.length; i++) {
        elem = arr[i];
        key = elem['name'];
        value = elem['value'];
        retval[key] = value;
    }
    return retval;
};


// class Page_GameInstructions2 {
//     constructor(parent_container, num) {
//         this.ID = `game_instructions_${num}`;
//         this.header = `Game ${num}/${total_games}`
//         this.info_div_width = "50%"
//         this.layout_display_div = "50%"
//         this.content_padding = "10px";
//         this.row_bg = ['#F5F5F5','white']
//         this.priming_options = [
//             "Take a direct route by carry objects through puddles",
//             "Take a longer detour around puddles",
//             "Pass objects to partner using counter tops"
//         ].sort(() => Math.random() - 0.5); // shuffle priming options
//
//         // Create Page Container
//         this.container = document.createElement("div");
//         this.container.style.display = "none";
//         this.container.style.width = "99%";
//         this.container.style.height = "99%";
//         // center content of container vertically
//         // this.container.style.display = "flex";
//         this.container.style.margin = "0 auto";
//         this.container.style.padding = "0px";
//
//         if (debug_mode) {
//             this.container.style.border = "2px solid blue";
//         }
//         parent_container.appendChild(this.container);
//
//         // Create game meta-data requested from server
//         this.layout = ['risky_tutorial_0','risky_coordination_ring'][num]
//         this.player1_name = 'StayAI'
//         this.player0_name = 'StayAI'
//         this.game_type = 'overcooked'
//         // this.layout = null;
//         this.p_slip = null;
//
//         this.add_submit_button('Begin Game');
//     }
//
//     render() {
//         // render game instructions
//         const header = document.createElement("h3");
//         header.innerText = this.header;
//         header.style.textAlign = "center";
//         header.style.margin = "20px";
//         this.container.appendChild(header);
//
//
//         const content_div = document.createElement("div");
//         content_div.width = "100%";
//         content_div.style.display = "flex";
//         this.container.appendChild(content_div);
//         content_div.style.margin = "0 auto";
//
//
//         // create game information text: ##################
//         const info_div = document.createElement("div");
//         info_div.style.width = this.info_div_width;
//         info_div.style.margin = "0 auto";
//         info_div.style.padding = this.content_padding;
//
//         const info_pslip = document.createElement("p");
//         info_pslip.innerHTML = `You have a ${this.p_slip}% chance of slipping in a puddle and losing your held object`;
//         info_div.appendChild(info_pslip);
//
//         const priming_text = document.createElement("p");
//         priming_text.innerHTML = "Before you begin, please select what you believe to be the best strategy in this game:"
//         priming_text.style.fontWeight = "bold";
//         info_div.appendChild(priming_text);
//
//         // create priming options radio button
//         const table = document.createElement("table");
//         table.style.border = "none";
//
//         this.priming_options.forEach((option, i) => {
//             const row = document.createElement("tr");
//             // make background white
//             row.style.backgroundColor = this.row_bg[i % 2];
//             row.style.padding = "2px";
//             table.appendChild(row)
//
//             const cell = document.createElement("td");
//             cell.style.padding = "2px";
//             cell.style.margin = "0";
//             cell.style.textAlign = "center";
//             cell.style.verticalAlign = "center";
//             cell.style.border = "none";
//             // cell.style.width = this.radio_cell_width;
//
//             const radio = document.createElement("input");
//             radio.type = "radio";
//             radio.name = `priming_${this.layout}`;
//             radio.value = option;
//             // align radio in cell
//             radio.style.margin = "0";
//             radio.style.padding = "0";
//             radio.style.verticalAlign = "center";
//             radio.style.textAlign = "left";
//             cell.appendChild(radio);
//             row.appendChild(cell);
//
//
//             const labelCell = document.createElement("td");
//             labelCell.innerText = option;
//             // labelCell.style.width = this.question_cell_width;
//             labelCell.style.padding = "2px";
//             labelCell.style.border = "none";
//             // labelCell.style.border = this.cell_border_style;
//             row.appendChild(labelCell);
//
//         });
//         info_div.appendChild(table);
//         content_div.appendChild(info_div);
//
//         // Render image of layout################################
//         const layout_div = document.createElement("div");
//         layout_div.id = "overcooked";
//         layout_div.style.width = this.layout_display_div;
//         layout_div.style.padding = this.content_padding;
//         layout_div.style.margin = "0 auto";
//         if (debug_mode) {
//             layout_div.style.border = "2px solid red";
//         }
//         // layout_div.style.margin = "0 auto";
//         this.render_layout(layout_div)
//         content_div.appendChild(layout_div);
//
//
//     };
//
//     render_layout(container) {
//         container.style.backgroundColor = "gray";
//         container.style.innerHTML = "Game Layout Picture";
//         alert("ERROR: ADD LAYOUT QUERY HERE");
//     }
//
//     emit_game_query() {
//         const params = {
//             'layouts': [this.layout],
//             phaseTwoScore: tutorial_config['tutorialParams']['phaseTwoScore'],
//             playerOne: this.player1_name,
//             playerZero: this.player0_name
//         }
//         // socket.emit('game_query', { 'ID' : this.ID });
//         const data = {
//             // "params" : config['tutorialParams'],
//             // "params" : tutorial_config['tutorialParams'],
//             "params" :params,
//             "game_name" : this.game_type
//         };
//         // create (or join if it exists) new game
//         socket.emit("join", data);
//     }
//
//     emit_responses(response) {
//         // sends priming response to server
//         console.log(response);
//     }
//
//     isFormComplete() {
//         if (debug_mode) {
//             return true;
//         }
//         ;
//         // verify that a radio option is selected
//         const selected = document.querySelector(`input[name="priming_${this.layout}"]:checked`);
//         return selected;
//     }
//
//     collectResponses() {
//         // get the values of the radio button questions
//         const responses = {"ID": this.ID};
//         if (debug_mode) {
//             responses["priming"] = "debug";
//         } else {
//             const selected = document.querySelector(`input[name="priming_${this.layout}"]:checked`);
//             responses["priming"] = selected.value;
//         }
//         return responses;
//     }
//
//     add_submit_button(txt) {
//         // Add submit button for custom event
//         this.submitBtn = document.createElement("button");
//         this.submitBtn.id = "begin-game-btn";
//         this.submitBtn.innerText = txt;
//         this.submitBtn.style.width = nextBtn.style.width;
//         this.submitBtn.style.height = nextBtn.style.height;
//         this.submitBtn.style.display = "none";
//         this.submitBtn.addEventListener("click", () => {
//             this.submit();
//         });
//         buttonContainer.appendChild(this.submitBtn);
//     }
//
//     submit() {
//         if (!this.isFormComplete()) {
//             alert("Please answer all questions before submitting.");
//             return;
//         }
//         // get the values of the questions
//         this.emit_responses(this.collectResponses());
//         this.destroy()
//         currentIndex++;
//         updatePage();
//         // this.destroy()
//
//     };
//
//     show() {
//         this.emit_game_query();
//         this.render()
//         this.container.style.display = "block";
//         this.submitBtn.style.display = "block";
//         nextBtn.style.display = "none";
//         prevBtn.style.display = "none";
//         // this.emit_game_query()
//         // this.render()
//     }
//
//     hide() {
//         this.container.style.display = "none";
//         this.submitBtn.style.display = "none";
//         // this.destroy()
//     }
//
//     destroy() {
//         // destroys game to save on memory
//         // alert("ERROR: ADD DESTROY GAME QUERY HERE");
//         this.container.innerHTML = "";
//     }
// }
//
//
// class Page_Tutorials2 {
//     constructor(parent_container, num) {
//         this.ID = `tutorial-${num}`;
//         this.layout = tutorial_config['tutorialParams']['layouts'][num]
//         this.p_slip = [0.0,0.3,0.9,0.9][num]
//         this.header = [
//             'Tutorial 1/4: Controls and Objective',
//             'Tutorial 2/4: Puddles',
//             'Tutorial 3/4: Taking a Detour',
//             'Tutorial 3/4: Passing Objects'
//         ][num]
//         this.AIname = ["TutorialAI","TutorialAI","TutorialAI"][num]
//         this.instuctions_txt = this.get_instructions()[num]
//         this.instuctions_fontsz = "14px";
//
//
//         // Create Page Container
//         this.container = document.createElement("div");
//         this.container.style.display = "none";
//         this.container.id = `tutorial_${num}`;
//         if (debug_mode) {
//             this.container.style.border = "2px solid red";
//         }
//         parent_container.appendChild(this.container);
//         this.add_submit_button('Continue')
//
//     }
//     game_finished() {
//         alert("ERROR: ADD GAME FINISHED QUERY HERE");
//         graphics_end();
//         disable_key_listener();
//         this.submitBtn.style.display = "block";
//     }
//
//     get_instructions() {
//         const instruct = [
//         `
//         <p>Your goal here is to cook and deliver onion soup in order to earn reward.</p>
//         <p>Use the <b>arrow keys</b> to move and <b>spacebar</b> to interact with objects</p>
//         <p>See if you can copy his actions in order to cook and deliver the appropriate soup</p>
//         <p>You need to <b>Place 3 onions in pot >> Wait for soup to cook >> Bring a Dish to the pot >> Deliver to service window</b></p>
//          <p>You will advance when you have delivered a soup</p>
//         `,
//         `
//         <p>Oh no! Someone spilled water on the floor and created a slippery puddle!</p>
//         <p>If you enter a paddle while holding an object, you may slip and fall.</p>
//         <p>When you slip, you will <b>lose your held object</b>.</p>
//         <p>Each game will have a <b>different chance of slipping</b>. Here, it is a 50% chance.</p>
//         <p>Try and cook a soup by navigating through the puddle.</p>
//          <p>You will advance when you have delivered a soup</p>
//         `,
//         `
//         <p>One option you have is to simply go around the puddles.</p>
//         <p>You must decide which is better:</p>
//          <ul>
//                     <li>Taking a longer path to avoid the puddle</li>
//                     <li>Walking through the puddle and risking a slip</li>
//            </ul>
//         <p>In this game, there is a ${this.p_slip}% chance of slipping.</p>
//          <p>You will advance when you have delivered a soup</p>
//         `,
//         `
//         <p>Alternatively, you can rely on your partner and <b>pass objects over the counter</b>.</p>
//         <p>In this game, there is a ${this.p_slip}% chance of slipping.</p>
//         <p>Therefore, the only reasonable stratagy is to pass items to your partner.</p>
//         <p>Try picking up onions/dishes and putting them on the counter where your partner can reach.</p>
//         <p>Your partner will then pass a soup to you for delivery.</p>
//         <p>You will advance when you have delivered a soup</p>
//         `
//     ];
//         return instruct
//
//
//
//     }
//
//     render() {
//         const header = document.createElement("h3");
//         header.innerText = this.header;
//         header.style.textAlign = "center";
//         header.style.margin = "20px";
//         header.style.width = "100%";
//         header.style.height = "10%";
//         this.container.appendChild(header);
//
//         const instructions = document.createElement("div");
//         instructions.id = "instructions";
//         instructions.style.width = "100%";
//         instructions.style.height = "25%";
//         instructions.innerHTML = this.instuctions_txt;
//         instructions.style.fontSize = this.instuctions_fontsz;
//         this.container.appendChild(instructions);
//
//         const overcooked = document.createElement("div");
//         overcooked.id = "overcooked";
//         overcooked.style.width = "100%";
//         overcooked.style.height = "70%";
//         overcooked.style.margin = "0 auto";
//         overcooked.style.padding = "0";
//         this.container.appendChild(overcooked);
//
//     }
//
//     emit_game_query() {
//
//         const tut_params = {
//             'layouts': [this.layout],
//             phaseTwoScore: tutorial_config['tutorialParams']['phaseTwoScore'],
//             playerOne: this.AIname,
//             playerZero: "human"
//         }
//         // socket.emit('game_query', { 'ID' : this.ID });
//         const data = {
//             // "params" : config['tutorialParams'],
//             // "params" : tutorial_config['tutorialParams'],
//             "params" : tut_params,
//             "game_name" : "tutorial"
//         };
//         // create (or join if it exists) new game
//         socket.emit("join", data);
//     }
//
//     submit() {
//         this.destroy()
//         currentIndex++;
//         updatePage();
//     };
//
//     add_submit_button(txt) {
//         // Add submit button for custom event
//         this.submitBtn = document.createElement("button");
//         this.submitBtn.id = `${this.ID}-btn`;
//         this.submitBtn.innerText = txt;
//         this.submitBtn.style.width = nextBtn.style.width;
//         this.submitBtn.style.height = nextBtn.style.height;
//         this.submitBtn.style.display = "none";
//         this.submitBtn.addEventListener("click", () => {
//             this.submit();
//         });
//         buttonContainer.appendChild(this.submitBtn);
//     }
//
//     show() {
//         this.emit_game_query();
//         this.render()
//         // make items in this.container alighn horizontally and stack vertically
//         this.container.style.display = "block";
//         this.container.style.flexDirection = "column";
//         this.container.style.alignItems = "center";
//
//         // this.container.style.display = "flex";
//         nextBtn.style.display = "none";
//         prevBtn.style.display = "none";
//         this.submitBtn.style.display = "none";
//     }
//
//     hide() {
//         this.container.style.display = "none";
//         this.submitBtn.style.display = "none";
//     }
//
//     destroy() {
//         // destroys game to save on memory
//         // alert("ERROR: ADD DESTROY GAME QUERY HERE");
//         this.container.innerHTML = "";
//     }
// }
//
