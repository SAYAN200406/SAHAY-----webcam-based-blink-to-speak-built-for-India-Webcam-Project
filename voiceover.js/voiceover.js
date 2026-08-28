function speak(text) {
    const speech = new SpeechSynthesisUtterance(text);
    speech.rate = 1;
    window.speechSynthesis.speak(speech);
}

async function handleButtonClick(buttonName) {
    // 1. Speak the text out loud
    speak(buttonName + " clicked");

    // 2. Special handling if the EMERGENCY button is selected
    if (buttonName.toUpperCase() === 'EMERGENCY') {
        try {
            const response = await fetch("/emergency", { method: "POST" });
            const data = await response.json();
            
            const statusText = document.getElementById("statusText");
            const statusDot = document.getElementById("statusDot");
            
            if (statusText) statusText.textContent = data.message;
            if (statusDot) statusDot.style.color = "orange";
        } catch (error) {
            console.error("Emergency Alert Error:", error);
        }
    }
}