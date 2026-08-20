function speak(text) {
    const speech = new SpeechSynthesisUtterance(text);
    speech.rate = 1;
    window.speechSynthesis.speak(speech);
}

function handleButtonClick(button) {
    speak(button.innerText);
    button.click();
}