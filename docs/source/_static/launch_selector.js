/* pimm - "Build your run command" interactive selector.
   PyTorch "Start Locally"-style pills that assemble the correct
   `pimm launch` / `pimm submit` command. Vanilla JS, no dependencies. */

(function () {
  "use strict";

  // -- Widget state ---------------------------------------------------------
  var state = {
    target: "local", // local | slurm
    nodes: "1",
    gpus: "1",
    jobType: "batch", // batch | interactive
    walltime: "02:00:00",
    chain: "off", // off | "4"
    config: "panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask",
    dryRun: false,
    runName: "",
  };

  // Row definitions: single-select pill rows.
  var ROWS = [
    {
      key: "target",
      label: "Run target",
      options: [
        { value: "local", text: "Local" },
        { value: "slurm", text: "Slurm cluster" },
      ],
    },
    {
      key: "nodes",
      label: "Nodes",
      options: ["1", "2", "4", "8"].map(function (v) {
        return { value: v, text: v };
      }),
    },
    {
      key: "gpus",
      label: "GPUs / node",
      options: ["1", "2", "4", "8"].map(function (v) {
        return { value: v, text: v };
      }),
    },
    {
      key: "jobType",
      label: "Job type",
      options: [
        { value: "batch", text: "Batch" },
        { value: "interactive", text: "Interactive" },
      ],
    },
    {
      key: "walltime",
      label: "Walltime",
      options: ["00:30:00", "02:00:00", "12:00:00"].map(function (v) {
        return { value: v, text: v };
      }),
    },
    {
      key: "chain",
      label: "Requeue chain",
      options: [
        { value: "off", text: "Off" },
        { value: "4", text: "4 jobs" },
      ],
    },
  ];

  function isSlurm() {
    return state.target === "slurm";
  }

  // Which rows/options are disabled given the current state.
  function rowDisabled(key) {
    if (key === "nodes") return !isSlurm(); // multi-node needs Slurm
    if (key === "jobType" || key === "walltime") return !isSlurm();
    if (key === "chain") return !isSlurm(); // batch requeue or chained interactive slots
    return false;
  }

  // -- Command assembly -----------------------------------------------------
  function buildCommandParts() {
    var parts = [];
    if (isSlurm()) {
      parts.push("pimm submit");
      parts.push("--site " + state.target);
      parts.push("--resources.nnodes " + state.nodes);
      parts.push("--resources.nproc-per-node " + state.gpus);
      parts.push("--resources.time " + state.walltime);
      if (state.jobType === "interactive") {
        parts.push("--interactive");
      }
      if (state.chain === "4") {
        // Batch requeues; interactive runs 4 sequential resuming salloc slots.
        parts.push("--chain.jobs 4");
      }
    } else {
      // Local: no --site, nodes always 1 (not emitted).
      parts.push("pimm launch");
      parts.push("--resources.nproc-per-node " + state.gpus);
    }

    if (state.runName.trim() !== "") {
      parts.push("--run.name " + state.runName.trim());
    }
    parts.push("--train.config " + state.config.trim());
    if (state.dryRun) {
      parts.push("--dry-run");
    }
    return parts;
  }

  function buildCommandText() {
    // Render on one logical line, wrapped with ` \` + newline for readability.
    return buildCommandParts().join(" \\\n  ");
  }

  // -- DOM construction -----------------------------------------------------
  var els = {}; // cache for live updates

  function makeRow(def) {
    var row = document.createElement("div");
    row.className = "pls-row";
    row.dataset.key = def.key;

    var label = document.createElement("div");
    label.className = "pls-label";
    label.textContent = def.label;
    row.appendChild(label);

    var opts = document.createElement("div");
    opts.className = "pls-options";

    def.options.forEach(function (opt) {
      var pill = document.createElement("button");
      pill.type = "button";
      pill.className = "pls-pill";
      pill.textContent = opt.text;
      pill.dataset.value = opt.value;
      pill.addEventListener("click", function () {
        if (rowDisabled(def.key)) return;
        state[def.key] = opt.value;
        onChange();
      });
      opts.appendChild(pill);
    });

    row.appendChild(opts);
    els[def.key] = row;
    return row;
  }

  function makeTextRow(label, key, placeholder, fullWidth) {
    var row = document.createElement("div");
    row.className = "pls-row";

    var lab = document.createElement("div");
    lab.className = "pls-label";
    lab.textContent = label;
    row.appendChild(lab);

    var wrap = document.createElement("div");
    wrap.className = "pls-options";

    var input = document.createElement("input");
    input.type = "text";
    input.className = "pls-input";
    input.value = state[key];
    if (placeholder) input.placeholder = placeholder;
    if (fullWidth) input.style.width = "100%";
    input.addEventListener("input", function () {
      state[key] = input.value;
      onChange();
    });
    wrap.appendChild(input);
    row.appendChild(wrap);
    return row;
  }

  function makeToggleRow() {
    var row = document.createElement("div");
    row.className = "pls-row";

    var lab = document.createElement("div");
    lab.className = "pls-label";
    lab.textContent = "Options";
    row.appendChild(lab);

    var wrap = document.createElement("div");
    wrap.className = "pls-options";

    // Dry run checkbox
    var dryLabel = document.createElement("label");
    dryLabel.className = "pls-toggle";
    var dry = document.createElement("input");
    dry.type = "checkbox";
    dry.checked = state.dryRun;
    dry.addEventListener("change", function () {
      state.dryRun = dry.checked;
      onChange();
    });
    dryLabel.appendChild(dry);
    dryLabel.appendChild(document.createTextNode("Dry run"));
    wrap.appendChild(dryLabel);

    // Run name input
    var nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "pls-input";
    nameInput.placeholder = "Run name (optional)";
    nameInput.value = state.runName;
    nameInput.style.flex = "0 1 16rem";
    nameInput.addEventListener("input", function () {
      state.runName = nameInput.value;
      onChange();
    });
    wrap.appendChild(nameInput);

    row.appendChild(wrap);
    return row;
  }

  function makeOutput() {
    var box = document.createElement("div");
    box.className = "pls-output";

    var copy = document.createElement("button");
    copy.type = "button";
    copy.className = "pls-copy";
    copy.textContent = "Copy";
    copy.addEventListener("click", function () {
      var text = buildCommandText();
      var done = function () {
        copy.textContent = "Copied!";
        copy.classList.add("is-copied");
        setTimeout(function () {
          copy.textContent = "Copy";
          copy.classList.remove("is-copied");
        }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try {
          document.execCommand("copy");
        } catch (e) {}
        document.body.removeChild(ta);
        done();
      }
    });
    box.appendChild(copy);

    var pre = document.createElement("pre");
    pre.className = "pls-command";
    els.command = pre;
    box.appendChild(pre);

    var hint = document.createElement("p");
    hint.className = "pls-hint";
    hint.innerHTML =
      "Tip: add <code>--dry-run</code> to preview the rendered " +
      "script/manifest before submitting.";
    box.appendChild(hint);

    return box;
  }

  // -- Live refresh ---------------------------------------------------------
  function onChange() {
    // Enforce invariants before rendering.
    if (!isSlurm()) {
      state.nodes = "1"; // Local is always single-node.
    }
    if (!isSlurm() || state.jobType === "interactive") {
      state.chain = "off"; // chain invalid with Interactive / Local.
    }
    refresh();
  }

  function refresh() {
    // Update pill selection + disabled rows.
    ROWS.forEach(function (def) {
      var row = els[def.key];
      var disabled = rowDisabled(def.key);
      row.classList.toggle("is-disabled", disabled);
      var pills = row.querySelectorAll(".pls-pill");
      pills.forEach(function (pill) {
        pill.classList.toggle(
          "is-selected",
          pill.dataset.value === state[def.key]
        );
        pill.classList.toggle("is-disabled", disabled);
        pill.disabled = disabled;
      });
    });
    // Update command box.
    els.command.textContent = buildCommandText();
  }

  function build(mount) {
    mount.innerHTML = "";
    var card = document.createElement("div");
    card.className = "pls-card";

    ROWS.forEach(function (def) {
      card.appendChild(makeRow(def));
    });
    card.appendChild(makeTextRow("Config", "config", "", true));
    card.appendChild(makeToggleRow());
    card.appendChild(makeOutput());

    mount.appendChild(card);
    refresh();
  }

  function init() {
    var mount = document.getElementById("pimm-launch-selector");
    if (!mount) return;
    build(mount);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
