// Check link safety via FastAPI
async function checkLink(url) {
  try {
    const formData = new FormData();
    formData.append("url", url);

    const response = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.results && result.results.url_check) {
      const prediction = result.results.url_check.explanation[0] || "Unknown";
      const confidence = (result.results.url_check.score * 100).toFixed(2) || "0";

      if (prediction === "phishing") {
  showAlert(`⚠️ PHISHING DETECTED (Confidence: ${confidence}%)`, url);
} else {
  window.open(url, "_blank");
}
    } else {
      showAlert("Unable to analyze this URL.", url);
    }
  } catch (error) {
    console.error("Error checking link:", error);
    showAlert("Error connecting to API.", url);
  }
}

// Show warning popup
function showAlert(message, url) {
  const modal = document.getElementById("alertModal");
  const alertMessage = document.getElementById("alertMessage");
  alertMessage.textContent = message;
  modal.style.display = "block";

  document.querySelector(".close").onclick = () => modal.style.display = "none";
  document.getElementById("cancelButton").onclick = () => modal.style.display = "none";
  document.getElementById("proceedButton").onclick = () => {
    modal.style.display = "none";
    window.open(url, "_blank");
  };
}

// Attach click handler to all <a> links
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      checkLink(link.href);
    });
  });
});
