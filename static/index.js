function swapToTest() {
    document.getElementById("swapPart").style.visibility = "hidden";
    document.getElementById("testPart").style.visibility = "visible";
}
function swapToAbout() {
    document.getElementById("swapPart").style.visibility = "hidden";

    document.getElementById("aboutPart").style.visibility = "visible";
}
function backToMain() {
    document.getElementById("testPart").style.visibility = "hidden";
    document.getElementById("aboutPart").style.visibility = "hidden";
    document.getElementById("swapPart").style.visibility = "visible";
}