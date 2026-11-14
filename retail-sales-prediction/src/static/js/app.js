document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file");
    const fileNote = document.getElementById("file-name");
    if (fileInput && fileNote) {
        fileInput.addEventListener("change", function () {
            fileNote.textContent = fileInput.files.length
                ? fileInput.files[0].name
                : "No file selected";
        });
    }

    const overlay = document.getElementById("overlay");
    ["upload-form", "train-form"].forEach(function (id) {
        const form = document.getElementById(id);
        if (!form || !overlay) return;
        form.addEventListener("submit", function () {
            setTimeout(function () { overlay.classList.remove("hidden"); }, 30);
        });
    });
});
