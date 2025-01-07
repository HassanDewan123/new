document.getElementById("predictionForm").onsubmit = async function(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);

    const response = await fetch("/predict", {
        method: "POST",
        body: new URLSearchParams(data),
    });

    const result = await response.json();
    const resultDiv = document.getElementById("result");

    if (result.error) {
        resultDiv.textContent = "Error: " + result.error;
    } else {
        resultDiv.innerHTML = `<strong>${result.prediction}</strong> with a probability of ${result.probability}`;
    }
};
