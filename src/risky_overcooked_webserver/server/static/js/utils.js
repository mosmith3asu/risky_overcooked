
class MultiItemSurvey {
    constructor(containerId, questions) {
        this.container = document.getElementById(containerId);
        this.questions = questions;
        this.answers = {}; // Store user responses
        this.renderSurvey();
    }

    renderSurvey() {
        if (!this.container) {
            console.error("Container element not found");
            return;
        }

        const form = document.createElement("form");
        form.addEventListener("submit", (event) => this.handleSubmit(event));

        this.questions.forEach((question, index) => {
            const questionContainer = document.createElement("div");
            questionContainer.classList.add("question-container");

            const label = document.createElement("label");
            label.textContent = question;
            questionContainer.appendChild(label);

            ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"].forEach((option, optionIndex) => {
                const input = document.createElement("input");
                input.type = "radio";
                input.name = `question-${index}`;
                input.value = optionIndex + 1;
                input.addEventListener("change", () => this.handleInputChange(index, optionIndex + 1));

                const optionLabel = document.createElement("label");
                optionLabel.textContent = option;
                optionLabel.prepend(input);

                questionContainer.appendChild(optionLabel);
            });

            form.appendChild(questionContainer);
        });

        const submitButton = document.createElement("button");
        submitButton.type = "submit";
        submitButton.textContent = "Submit";
        form.appendChild(submitButton);

        this.container.appendChild(form);
    }

    handleInputChange(questionIndex, value) {
        this.answers[questionIndex] = value;
    }

    handleSubmit(event) {
        event.preventDefault();
        console.log("Survey responses:", this.answers);
        alert("Survey submitted! Check the console for responses.");
    }
}

 const survey = new MultiItemSurvey("survey-container", ["Question 1?", "Question 2?", "Question 3?"]);