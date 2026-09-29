/* =========================================================
   WASTEAI — FRONTEND JAVASCRIPT
   Connected to FastAPI + PyTorch
   ========================================================= */


/* =========================================================
   API CONFIGURATION
   ========================================================= */

const API_URL = "http://127.0.0.1:8000/predict";


/* =========================================================
   ELEMENTS
   ========================================================= */

const uploadArea =
    document.getElementById("uploadArea");

const uploadButton =
    document.getElementById("uploadButton");

const imageInput =
    document.getElementById("imageInput");

const imagePreview =
    document.getElementById("imagePreview");

const previewImage =
    document.getElementById("previewImage");

const fileName =
    document.getElementById("fileName");

const classifyButton =
    document.getElementById("classifyButton");

const predictionResult =
    document.getElementById("predictionResult");

const predictionClass =
    document.getElementById("predictionClass");

const confidenceFill =
    document.getElementById("confidenceFill");

const confidenceText =
    document.getElementById("confidenceText");

const disposalGuidance =
    document.getElementById("disposalGuidance");


/* =========================================================
   TOP 3 RESULT CONTAINER
   ========================================================= */

const topPredictionsContainer =
    document.createElement("div");

topPredictionsContainer.className =
    "top-predictions";

predictionResult.appendChild(
    topPredictionsContainer
);


/* =========================================================
   OPEN FILE SELECTOR
   ========================================================= */

uploadButton.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();

        imageInput.click();

    }
);


/* =========================================================
   UPLOAD AREA CLICK
   ========================================================= */

uploadArea.addEventListener(
    "click",
    function (event) {

        if (
            event.target === uploadButton
        ) {
            return;
        }

        imageInput.click();

    }
);


/* =========================================================
   FILE SELECTED
   ========================================================= */

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (!file) {
            return;
        }

        handleFile(file);

    }
);


/* =========================================================
   HANDLE FILE
   ========================================================= */

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];


    if (
        !allowedTypes.includes(
            file.type
        )
    ) {

        alert(
            "Please upload a JPG, JPEG or PNG image."
        );

        return;
    }


    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            previewImage.src =
                event.target.result;

            fileName.textContent =
                file.name;

            uploadArea.style.display =
                "none";

            imagePreview.style.display =
                "flex";

            predictionResult.style.display =
                "none";

            confidenceFill.style.width =
                "0%";

            topPredictionsContainer.innerHTML =
                "";

        };


    reader.readAsDataURL(file);

}


/* =========================================================
   DRAG OVER
   ========================================================= */

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.style.borderColor =
            "rgba(126, 231, 135, 0.8)";

        uploadArea.style.background =
            "rgba(126, 231, 135, 0.08)";

    }
);


/* =========================================================
   DRAG LEAVE
   ========================================================= */

uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.style.borderColor =
            "rgba(126, 231, 135, 0.28)";

        uploadArea.style.background =
            "";

    }
);


/* =========================================================
   DROP
   ========================================================= */

uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();


        uploadArea.style.borderColor =
            "rgba(126, 231, 135, 0.28)";

        uploadArea.style.background =
            "";


        const file =
            event.dataTransfer.files[0];


        if (!file) {
            return;
        }


        /*
           Put the dropped file into
           the file input so the
           classify button can access it.
        */

        const dataTransfer =
            new DataTransfer();

        dataTransfer.items.add(file);

        imageInput.files =
            dataTransfer.files;


        handleFile(file);

    }
);


/* =========================================================
   CLASSIFY BUTTON
   ========================================================= */

classifyButton.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];


        if (!file) {

            alert(
                "Please select an image first."
            );

            return;
        }


        /*
           Loading state
        */

        classifyButton.disabled =
            true;

        classifyButton.textContent =
            "Analyzing AI Model...";


        try {

            /* ---------------------------------------------
               Create form data
               --------------------------------------------- */

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );


            /* ---------------------------------------------
               Send image to FastAPI
               --------------------------------------------- */

            const response =
                await fetch(
                    API_URL,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            /* ---------------------------------------------
               Check API response
               --------------------------------------------- */

            if (!response.ok) {

                throw new Error(
                    `API Error: ${response.status}`
                );

            }


            const data =
                await response.json();


            /* ---------------------------------------------
               Display result
               --------------------------------------------- */

            displayPrediction(data);


        } catch (error) {

            console.error(
                "Prediction Error:",
                error
            );


            alert(
                "Unable to connect to the AI server.\n\n" +
                "Make sure FastAPI is running on:\n" +
                "http://127.0.0.1:8000"
            );


        } finally {

            classifyButton.disabled =
                false;

            classifyButton.textContent =
                "Analyze Again →";

        }

    }
);


/* =========================================================
   DISPLAY PREDICTION
   ========================================================= */

function displayPrediction(data) {

    /*
       Make sure API returned valid data
    */

    if (!data.success) {

        alert(
            "The AI server returned an invalid response."
        );

        return;
    }


    const predictedClass =
        data.class;

    const confidence =
        data.confidence;


    /* ---------------------------------------------
       Main prediction
       --------------------------------------------- */

    predictionClass.textContent =
        capitalize(
            predictedClass
        );


    /* ---------------------------------------------
       Confidence
       --------------------------------------------- */

    confidenceText.textContent =
        `${confidence.toFixed(2)}% confidence`;


    confidenceFill.style.width =
        `${confidence}%`;


    /* ---------------------------------------------
       Disposal guidance
       --------------------------------------------- */

    disposalGuidance.textContent =
        data.disposal_guidance;


    /* ---------------------------------------------
       Top 3 predictions
       --------------------------------------------- */

    displayTopPredictions(
        data.top_predictions
    );


    /* ---------------------------------------------
       Show result
       --------------------------------------------- */

    predictionResult.style.display =
        "block";


    /* ---------------------------------------------
       Scroll to result
       --------------------------------------------- */

    setTimeout(
        function () {

            predictionResult.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

        },
        150
    );

}


/* =========================================================
   TOP 3 PREDICTIONS
   ========================================================= */

function displayTopPredictions(
    predictions
) {

    topPredictionsContainer.innerHTML =
        "";


    if (
        !predictions ||
        predictions.length === 0
    ) {
        return;
    }


    const title =
        document.createElement("h4");

    title.textContent =
        "Top 3 Predictions";


    topPredictionsContainer.appendChild(
        title
    );


    predictions.forEach(
        function (prediction) {

            const item =
                document.createElement("div");

            item.className =
                "top-prediction";


            const name =
                document.createElement("span");

            name.textContent =
                capitalize(
                    prediction.class
                );


            const value =
                document.createElement("strong");

            value.textContent =
                `${prediction.confidence.toFixed(2)}%`;


            const bar =
                document.createElement("div");

            bar.className =
                "mini-confidence";


            const fill =
                document.createElement("div");

            fill.className =
                "mini-confidence-fill";


            fill.style.width =
                `${prediction.confidence}%`;


            bar.appendChild(
                fill
            );


            const row =
                document.createElement("div");

            row.className =
                "prediction-row";


            row.appendChild(
                name
            );

            row.appendChild(
                value
            );


            item.appendChild(
                row
            );

            item.appendChild(
                bar
            );


            topPredictionsContainer.appendChild(
                item
            );

        }
    );

}


/* =========================================================
   CAPITALIZE
   ========================================================= */

function capitalize(text) {

    return text.charAt(0).toUpperCase()
        + text.slice(1);

}