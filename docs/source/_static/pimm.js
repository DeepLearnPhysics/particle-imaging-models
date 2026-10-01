/* pimm docs: inline the themed logo on the landing page and draw the 3D event views. */
(function () {
  "use strict";
  var script = document.currentScript;
  var base = script ? script.src.replace(/pimm\.js(\?.*)?$/, "") : "_static/";
  var PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
  var plots = [];
  var loading = null;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  document.querySelectorAll(".hero-mark").forEach(function (el) {
    fetch(base + "logo-themed.svg")
      .then(function (r) { return r.text(); })
      .then(function (svg) { el.innerHTML = svg; })
      .catch(function () {});
  });

  function loadPlotly() {
    if (window.Plotly) return Promise.resolve();
    if (!loading) {
      loading = new Promise(function (resolve, reject) {
        var s = document.createElement("script");
        s.src = PLOTLY;
        s.async = true;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    return loading;
  }

  function axis(title) {
    return {
      title: { text: title },
      color: cssVar("--pst-color-text-muted"),
      gridcolor: cssVar("--pst-color-border"),
      zerolinecolor: cssVar("--pst-color-border"),
      showbackground: false,
      showspikes: false
    };
  }

  function layoutFor(fig) {
    return {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      font: { family: cssVar("--pst-font-family-base"), color: cssVar("--pst-color-text-base"), size: 13 },
      showlegend: true,
      legend: { orientation: "h", x: 0, y: 0.02, bgcolor: "rgba(0,0,0,0)" },
      scene: {
        xaxis: axis("x"), yaxis: axis("y"), zaxis: axis("z"),
        aspectmode: "data", bgcolor: "rgba(0,0,0,0)",
        camera: fig.layout.scene && fig.layout.scene.camera
      }
    };
  }

  document.querySelectorAll(".event-viewer[data-figure]").forEach(function (box) {
    box.innerHTML =
      '<div class="ev-bar" role="group" aria-label="Color points by">' +
      '<button type="button" aria-pressed="true" data-set="left">Semantic class</button>' +
      '<button type="button" aria-pressed="false" data-set="right">Particle class</button></div>' +
      '<div class="ev-plot"><p class="ev-status">Loading the 3D view…</p></div>';
    var plotEl = box.querySelector(".ev-plot");
    var figure = fetch(base + "figures/" + box.dataset.figure + ".json").then(function (r) { return r.json(); });
    Promise.all([figure, loadPlotly()]).then(function (res) {
      var fig = res[0];
      var traces = fig.data.map(function (t) {
        var c = JSON.parse(JSON.stringify(t));
        c.scene = "scene";
        c.visible = c.legendgroup.indexOf("left") === 0;
        c.marker.size = 3;
        return c;
      });
      plotEl.innerHTML = "";
      window.Plotly.newPlot(plotEl, traces, layoutFor(fig), { displayModeBar: false, responsive: true });
      plots.push({ el: plotEl, fig: fig });
      box.querySelectorAll(".ev-bar button").forEach(function (b) {
        b.addEventListener("click", function () {
          box.querySelectorAll(".ev-bar button").forEach(function (o) { o.setAttribute("aria-pressed", o === b); });
          window.Plotly.restyle(plotEl, {
            visible: traces.map(function (t) { return t.legendgroup.indexOf(b.dataset.set) === 0; })
          });
        });
      });
    }).catch(function () {
      plotEl.querySelector(".ev-status").textContent = "The 3D view needs to load Plotly from cdn.jsdelivr.net.";
    });
  });

  new MutationObserver(function () {
    plots.forEach(function (p) { window.Plotly.relayout(p.el, layoutFor(p.fig)); });
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
