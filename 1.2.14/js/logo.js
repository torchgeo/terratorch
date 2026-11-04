document.addEventListener("DOMContentLoaded", function () {
  const topic = document.querySelector(".md-header__title .md-header__topic:first-child");
  if (topic) {
    topic.innerHTML = '<img src="figs/logo_header.png" alt="TerraTorch Logo" style="height: 1.2rem; vertical-align: middle; display: inline-block; margin-top: 0.6rem;">';
  }
});
