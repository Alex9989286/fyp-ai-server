async function uploadAudio() {

    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an audio file");
        return;
    }

    document.getElementById("result").innerText = "Processing... ⏳";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("https://fyp-ai-server-1-v11.onrender.com/detect_sound", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            document.getElementById("result").innerText =
                "Error: " + data.error;
            return;
        }

        document.getElementById("result").innerText =
            `Result: ${data.label} (${data.confidence.toFixed(2)})`;

    } catch (error) {
        document.getElementById("result").innerText =
            "Request failed: " + error;
    }
}
