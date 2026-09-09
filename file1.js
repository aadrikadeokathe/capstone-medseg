const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5

const C = {
  bg: "0A1628",
  teal: "00B4A6",
  tealDark: "007A70",
  tealLight: "E0F7F5",
  amber: "FFB547",
  white: "FFFFFF",
  offWhite: "E8EDF2",
  grey: "8A9BB0",
  darkCard: "0F2240",
  midCard: "142D50",
  red: "E05C5C",
  green: "52C97A",
};

const TITLE_FONT = "Calibri";
const BODY_FONT = "Calibri";

function slide() {
  const s = pres.addSlide();
  s.background = { color: C.bg };
  return s;
}

function addLabel(s, text, x, y, w, h, opts = {}) {
  s.addText(text, {
    x, y, w, h,
    fontFace: opts.font || BODY_FONT,
    fontSize: opts.size || 11,
    color: opts.color || C.white,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || "left",
    valign: opts.valign || "middle",
    isTextBox: true,
    margin: 0,
    ...opts,
  });
}

function card(s, x, y, w, h, fillColor) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: fillColor || C.darkCard },
    line: { color: C.tealDark, width: 0.8 },
    rectRadius: 0.06,
  });
}

function slideHeader(s, kicker, title) {
  addLabel(s, kicker.toUpperCase(), 0.5, 0.22, 12, 0.3, { color: C.teal, size: 10, bold: true, charSpacing: 3 });
  addLabel(s, title, 0.5, 0.5, 12, 0.65, { size: 28, bold: true, color: C.white });
  s.addShape(pres.ShapeType.line, { x: 0.5, y: 1.22, w: 12.33, h: 0, line: { color: C.tealDark, width: 1 } });
}

function pageNum(s, n) {
  addLabel(s, `${n} / 10`, 12.5, 7.1, 0.75, 0.3, { color: C.grey, size: 9, align: "right" });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 1 — TITLE
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 3.5, h: 7.5, fill: { color: C.darkCard }, line: { type: "none" } });
  s.addShape(pres.ShapeType.rect, { x: 3.5, y: 0, w: 0.04, h: 7.5, fill: { color: C.teal }, line: { type: "none" } });

  addLabel(s, "NMIMS · MPSTME INDORE", 0.45, 0.38, 2.8, 0.3, { color: C.teal, size: 8.5, bold: true, charSpacing: 2, align: "center" });
  addLabel(s, "CAPSTONE PROJECT", 0.45, 0.7, 2.8, 0.28, { color: C.grey, size: 9, align: "center" });
  addLabel(s, "REVIEW 2", 0.45, 1.0, 2.8, 0.28, { color: C.amber, size: 9, bold: true, align: "center" });

  addLabel(s, "Team", 0.45, 5.5, 2.8, 0.28, { color: C.grey, size: 9, align: "center" });
  addLabel(s, "Aadrika Deokathe", 0.45, 5.8, 2.8, 0.28, { color: C.white, size: 10, bold: true, align: "center" });
  addLabel(s, "Khushi Chadda", 0.45, 6.1, 2.8, 0.28, { color: C.white, size: 10, align: "center" });
  addLabel(s, "Harsh Chhatri", 0.45, 6.35, 2.8, 0.28, { color: C.white, size: 10, align: "center" });
  addLabel(s, "Mentor: Dr. Shweta Gangrade", 0.45, 6.8, 2.8, 0.28, { color: C.teal, size: 9, align: "center" });

  addLabel(s, "Closing the Domain Gap:", 3.8, 1.0, 9.2, 0.9, { size: 36, bold: true, color: C.teal, font: TITLE_FONT });
  addLabel(s, "In-Context Learning for\nZero-Shot Multi-Modal\nMedical Segmentation", 3.8, 1.85, 9.2, 2.8, { size: 34, bold: true, color: C.white, font: TITLE_FONT, lineSpacingMultiple: 1.1 });
  addLabel(s, "A lightweight fusion adapter for generalizable medical image segmentation\nacross CT and MRI modalities without organ-specific retraining.", 3.8, 4.75, 9.0, 0.9, { size: 13, color: C.offWhite, italic: true });
  addLabel(s, "September 2026", 3.8, 6.85, 4, 0.3, { color: C.grey, size: 10 });
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 2 — PROBLEM STATEMENT
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 2", "Problem Statement");

  const problems = [
    { icon: "🏥", head: "One AI Per Organ", body: "Hospitals need a separate model for every organ, disease type, and scanner brand — expensive, slow, doesn't scale." },
    { icon: "🌐", head: "Domain Shift", body: "SAM (Meta's 'Segment Anything') works on photos but fails on CT/MRI — medical pixels look nothing like natural images." },
    { icon: "⚠️", head: "Catastrophic Forgetting", body: "Training a model on kidneys makes it forget lungs. Specialist models can't adapt to new tasks without full retraining." },
  ];
  problems.forEach((p, i) => {
    const x = 0.5 + i * 4.28;
    card(s, x, 1.45, 3.95, 2.8);
    addLabel(s, p.icon, x + 0.2, 1.6, 0.6, 0.6, { size: 22 });
    addLabel(s, p.head, x + 0.2, 2.22, 3.5, 0.38, { size: 14, bold: true, color: C.teal });
    addLabel(s, p.body, x + 0.2, 2.62, 3.6, 1.5, { size: 11, color: C.offWhite, lineSpacingMultiple: 1.25 });
  });

  card(s, 0.5, 4.5, 12.33, 1.6, C.midCard);
  addLabel(s, "Our Research Question", 0.8, 4.65, 6, 0.38, { size: 13, bold: true, color: C.amber });
  addLabel(s, "Can a small, trainable adapter on top of a frozen foundation model learn to segment a\nbrand-new organ from just 1-2 example images — without any retraining?", 0.8, 5.05, 11.8, 0.9, { size: 14, color: C.white, italic: true, lineSpacingMultiple: 1.3 });

  pageNum(s, 2);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 3 — LITERATURE REVIEW
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 3", "Literature Review");

  const headerOpts = { fill: { color: C.tealDark }, color: C.white, bold: true, fontFace: BODY_FONT, fontSize: 10, valign: "middle" };
  const cell = (text, opts = {}) => ({ text, options: { fontFace: BODY_FONT, fontSize: 9.5, valign: "middle", fill: { color: opts.alt ? "0F2240" : "0A1E36" }, color: C.offWhite, ...opts } });
  const hCell = (text) => ({ text, options: headerOpts });

  const rows = [
    [hCell("Paper / Year"), hCell("Core Idea"), hCell("What It Lacks"), hCell("Our Gap")],
    [cell("SAM (2023)"), cell("Segment anything in natural photos via click/box"), cell("Fails on medical images — domain shift"), cell("Needs medical adaptation")],
    [cell("MedSAM (2024)", { alt: true }), cell("SAM fine-tuned on 1.57M medical images", { alt: true }), cell("Needs manual box per image — not scalable", { alt: true }), cell("Example-based, not box-based", { alt: true })],
    [cell("UniverSeg (2023)"), cell("Segments from 1-2 examples, no retraining"), cell("One structure per pass, re-encodes every time"), cell("Multi-organ, cross-modality study")],
    [cell("Onco-Seg/SAM3 (2026)", { alt: true }), cell("Text prompt: 'find the tumor'", { alt: true }), cell("Only oncology, not general anatomy", { alt: true }), cell("General anatomy + MRI + CT", { alt: true })],
    [cell("Show & Segment (2025)"), cell("Multi-class in one pass, efficient"), cell("Lower accuracy vs specialists"), cell("Quantified transfer failure analysis")],
  ];

  s.addTable(rows, {
    x: 0.5, y: 1.4, w: 12.35, h: 5.7,
    colW: [2.3, 3.2, 3.1, 3.75],
    border: { type: "solid", color: "1A3A5C", pt: 0.5 },
    autoPage: false,
  });

  pageNum(s, 3);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 4 — SYSTEM ARCHITECTURE
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 4", "System Architecture");

  const stages = [
    { title: "INPUT", sub: "New scan +\n1-2 example slices\n(CT or MRI)", color: C.midCard, badge: null },
    { title: "FROZEN BACKBONE", sub: "UniverSeg Encoder\n1,182,785 params\n77% of total · NEVER trained", color: "0D2B4E", badge: "FROZEN" },
    { title: "FUSION MODULE", sub: "3× Cross-Attention Blocks\nMask-gated attention\n355,457 params · 23%", color: "0A3540", badge: "TRAINED" },
    { title: "DECODER", sub: "Lightweight ConvNet\nUpsamples to original\nresolution", color: C.midCard, badge: "TRAINED" },
    { title: "OUTPUT", sub: "Pixel-level\nsegmentation mask", color: C.midCard, badge: null },
  ];

  const bw = 2.1, gap = 0.28, startX = 0.35, y = 1.5, h = 3.2;
  stages.forEach((st, i) => {
    const x = startX + i * (bw + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w: bw, h, fill: { color: st.color }, line: { color: C.teal, width: 1.2 }, rectRadius: 0.07 });
    addLabel(s, st.title, x + 0.1, y + 0.2, bw - 0.2, 0.4, { size: 11, bold: true, color: C.teal, align: "center" });
    addLabel(s, st.sub, x + 0.1, y + 0.65, bw - 0.2, 2.3, { size: 10, color: C.offWhite, align: "center", lineSpacingMultiple: 1.25 });
    if (st.badge) {
      const bc = st.badge === "FROZEN" ? C.red : C.green;
      s.addShape(pres.ShapeType.roundRect, { x: x + 0.35, y: y + h - 0.52, w: bw - 0.7, h: 0.38, fill: { color: bc, transparency: 80 }, line: { color: bc, width: 1 }, rectRadius: 0.04 });
      addLabel(s, st.badge, x + 0.35, y + h - 0.52, bw - 0.7, 0.38, { size: 9, bold: true, color: bc, align: "center" });
    }
    if (i < stages.length - 1) {
      addLabel(s, "→", x + bw + 0.02, y + h / 2 - 0.25, gap + 0.2, 0.5, { size: 18, bold: true, color: C.teal, align: "center" });
    }
  });

  const stats = [
    { label: "Total Parameters", val: "1,538,242" },
    { label: "Frozen (Backbone)", val: "1,182,785 · 77%" },
    { label: "Trainable (Fusion+Decoder)", val: "355,457 · 23%" },
    { label: "Backbone", val: "UniverSeg (pretrained)" },
  ];
  const sw = 2.9;
  stats.forEach((st, i) => {
    const x = 0.35 + i * (sw + 0.3);
    card(s, x, 5.0, sw, 1.25, C.darkCard);
    addLabel(s, st.label, x + 0.18, 5.12, sw - 0.3, 0.35, { size: 9, color: C.grey });
    addLabel(s, st.val, x + 0.18, 5.48, sw - 0.3, 0.55, { size: 12, bold: true, color: C.amber });
  });

  pageNum(s, 4);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 5 — DATASETS
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 5", "Datasets — Medical Segmentation Decathlon");

  const datasets = [
    { name: "Spleen", mod: "CT", slices: "305", split: "70/15/15", base: "0.5581", color: "1A6B8A" },
    { name: "Liver", mod: "CT", slices: "57*", split: "60/20/20", base: "0.4953", color: "1A6B8A" },
    { name: "Heart", mod: "MRI", slices: "1,301", split: "70/15/15", base: "0.3115", color: "7A3B8A" },
    { name: "Brain Tumour", mod: "MRI (FLAIR)", slices: "31,527", split: "70/15/15", base: "0.1614", color: "7A3B8A" },
  ];

  datasets.forEach((d, i) => {
    const x = 0.5 + i * 3.1;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.45, w: 2.88, h: 4.2, fill: { color: d.color, transparency: 85 }, line: { color: d.color, width: 1.2 }, rectRadius: 0.07 });
    s.addShape(pres.ShapeType.roundRect, { x: x + 0.15, y: 1.55, w: 2.58, h: 0.45, fill: { color: d.color }, line: { type: "none" }, rectRadius: 0.04 });
    addLabel(s, d.name, x + 0.15, 1.55, 2.58, 0.45, { size: 13, bold: true, color: C.white, align: "center" });
    const rows = [["Modality", d.mod], ["2D Slices", d.slices], ["Split", d.split], ["Baseline Dice", d.base]];
    rows.forEach((r, j) => {
      addLabel(s, r[0], x + 0.2, 2.18 + j * 0.72, 1.1, 0.38, { size: 10, color: C.grey });
      addLabel(s, r[1], x + 1.35, 2.18 + j * 0.72, 1.4, 0.38, { size: 11, bold: true, color: C.white, align: "right" });
    });
  });

  addLabel(s, "* Liver uses 60/20/20 split due to small volume count (57 slices from MSD Task03 subset). All datasets sourced from the Medical Segmentation Decathlon — publicly available, pre-labeled.", 0.5, 5.85, 12.35, 0.5, { size: 9, color: C.grey, italic: true });

  pageNum(s, 5);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 6 — RESULTS (MAIN)
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 6 — Implementation & Results", "Trained Results vs. Baseline");

  const datasets = ["Spleen\n(CT)", "Liver\n(CT)", "Heart\n(MRI)", "Brain Tumour\n(MRI)"];
  const baseline = [0.5581, 0.4953, 0.3115, 0.1614];
  const trained = [0.8884, 0.9341, 0.8300, 0.7524];
  const nnunet = [0.9130, null, null, null];

  s.addChart(pres.ChartType.bar, [
    { name: "UniverSeg Baseline (untrained)", labels: datasets, values: baseline },
    { name: "Our 3-Layer FusionModule (trained)", labels: datasets, values: trained },
    { name: "nnU-Net Specialist (Spleen only)", labels: datasets, values: nnunet },
  ], {
    x: 0.5, y: 1.4, w: 7.8, h: 5.6,
    barGrouping: "clustered",
    barGapWidthPct: 55,
    chartColors: ["4A6FA5", "00B4A6", "FFB547"],
    showTitle: false,
    showLegend: true,
    legendPos: "b",
    legendFontSize: 10,
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 9,
    dataLabelFontBold: true,
    dataLabelColor: "FFFFFF",
    valAxisMinVal: 0,
    valAxisMaxVal: 1.0,
    valAxisLabelColor: "8A9BB0",
    catAxisLabelColor: "E8EDF2",
    catAxisLabelFontSize: 10,
    valGridLine: { color: "1A3A5C", size: 0.8 },
  });

  const findings = [
    { n: "0.9341", label: "Best result\n(Liver CT)" },
    { n: "0.9130", label: "nnU-Net\nspecialist" },
    { n: "+59pp", label: "Biggest gain\n(Brain Tumour)" },
    { n: "23%", label: "Trainable\nparams only" },
  ];
  findings.forEach((f, i) => {
    card(s, 8.55, 1.4 + i * 1.35, 4.28, 1.22);
    addLabel(s, f.n, 8.75, 1.52 + i * 1.35, 2, 0.55, { size: 26, bold: true, color: C.teal });
    addLabel(s, f.label, 10.8, 1.52 + i * 1.35, 1.9, 0.55, { size: 10, color: C.offWhite, lineSpacingMultiple: 1.2 });
  });

  pageNum(s, 6);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 7 — DEPTH ABLATION
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 7", "Architecture Ablation — Fusion Module Depth");

  s.addChart(pres.ChartType.bar, [
    { name: "Test Dice Score", labels: ["1 Layer\n(288K params)", "3 Layers\n(355K params)", "5 Layers\n(423K params)"], values: [0.7489, 0.7765, 0.7758] },
  ], {
    x: 0.5, y: 1.45, w: 6.5, h: 5.2,
    barGrouping: "clustered",
    chartColors: ["00B4A6"],
    showTitle: false,
    showValue: true,
    dataLabelPosition: "outEnd",
    dataLabelFontSize: 12,
    dataLabelFontBold: true,
    dataLabelColor: "FFFFFF",
    valAxisMinVal: 0.6,
    valAxisMaxVal: 0.9,
    valAxisLabelColor: "8A9BB0",
    catAxisLabelColor: "E8EDF2",
    catAxisLabelFontSize: 10,
    valGridLine: { color: "1A3A5C" },
    showLegend: false,
  });

  const findings = [
    { head: "1 Layer (Baseline)", body: "0.7489 Test Dice\n19.6% trainable params\nFast, works, but limited feature refinement." },
    { head: "3 Layers (Optimal ✓)", body: "0.7765 Test Dice (+2.76%)\n23.1% trainable params\nBest balance: capacity vs. efficiency." },
    { head: "5 Layers (Plateau)", body: "0.7758 Test Dice\n26.3% trainable params\nDiminishing returns — confirms 3 is optimal." },
  ];
  findings.forEach((f, i) => {
    const isOpt = i === 1;
    card(s, 7.3, 1.45 + i * 1.72, 5.53, 1.58, C.darkCard);
    if (isOpt) s.addShape(pres.ShapeType.roundRect, { x: 7.3, y: 1.45 + i * 1.72, w: 5.53, h: 1.58, fill: { color: C.teal, transparency: 90 }, line: { color: C.teal, width: 2 }, rectRadius: 0.06 });
    addLabel(s, f.head, 7.55, 1.58 + i * 1.72, 5, 0.38, { size: 12, bold: true, color: isOpt ? C.teal : C.offWhite });
    addLabel(s, f.body, 7.55, 1.95 + i * 1.72, 5, 1.0, { size: 10.5, color: C.offWhite, lineSpacingMultiple: 1.3 });
  });

  pageNum(s, 7);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 8 — ZERO-SHOT ANALYSIS
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 8", "Zero-Shot Transfer Analysis — A Key Finding");

  const datasets = ["Spleen", "Liver", "Heart", "Brain Tumour"];
  const baseline = [0.5581, 0.4953, 0.3115, 0.1614];
  const zeroshot = [0.0147, 0.0104, 0.1068, 0.0278];
  const trained = [0.8884, 0.9341, 0.83, 0.7524];

  s.addChart(pres.ChartType.bar, [
    { name: "UniverSeg Baseline", labels: datasets, values: baseline },
    { name: "Zero-Shot (train on 3, test on 1 unseen)", labels: datasets, values: zeroshot },
    { name: "Per-Dataset Trained (upper bound)", labels: datasets, values: trained },
  ], {
    x: 0.5, y: 1.45, w: 7.6, h: 4.4,
    barGrouping: "clustered",
    chartColors: ["4A6FA5", "E05C5C", "00B4A6"],
    showTitle: false, showLegend: true, legendPos: "b", legendFontSize: 9,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 8,
    dataLabelColor: "FFFFFF",
    valAxisMinVal: 0, valAxisMaxVal: 1.0,
    valAxisLabelColor: "8A9BB0", catAxisLabelColor: "E8EDF2",
    valGridLine: { color: "1A3A5C" }, showLegend: true,
  });

  const insights = [
    { head: "What happened", body: "Training on 3 organs combined and testing on the 4th unseen organ scores near-zero across all 4 directions — worse than the untrained baseline." },
    { head: "Why this matters", body: "The FusionModule learns anatomy-specific features, not transferable 'how to segment from examples' representations. This is a systematic finding, not a bug." },
    { head: "The fix — in progress", body: "Episodic/meta-learning training: randomly alternate organs during training, forcing the module to learn generalizable features. Running now on GPU." },
  ];
  insights.forEach((ins, i) => {
    card(s, 8.35, 1.45 + i * 1.6, 4.48, 1.48);
    addLabel(s, ins.head, 8.55, 1.58 + i * 1.6, 4.1, 0.35, { size: 11, bold: true, color: C.amber });
    addLabel(s, ins.body, 8.55, 1.95 + i * 1.6, 4.1, 0.9, { size: 9.5, color: C.offWhite, lineSpacingMultiple: 1.2 });
  });

  pageNum(s, 8);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 9 — CONCLUSION & NEXT STEPS
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 9", "Conclusion & Next Steps");

  const done = [
    "4-dataset pipeline: CT (Spleen, Liver) + MRI (Heart, Brain Tumour)",
    "3-layer FusionModule confirmed optimal via ablation study",
    "Trained results: 0.88–0.93 Dice, consistent improvement across all datasets",
    "nnU-Net specialist baseline: 0.9130 (our adapter at 97% of specialist performance)",
    "Complete leave-one-out zero-shot transfer matrix — systematic failure characterized",
  ];
  const next = [
    "Episodic meta-learning training to fix zero-shot transfer",
    "Shot-count ablation: 1, 2, 4, 8 examples vs Dice curve",
    "Attention map visualizations for paper figures",
    "Full statistical evaluation with confidence intervals",
    "Target: IEEE BIBM 2026 workshop / arXiv preprint",
  ];

  card(s, 0.5, 1.45, 6.0, 5.7, C.darkCard);
  addLabel(s, "✅  Completed", 0.75, 1.6, 5.5, 0.38, { size: 13, bold: true, color: C.green });
  done.forEach((d, i) => {
    addLabel(s, `• ${d}`, 0.75, 2.08 + i * 0.95, 5.5, 0.85, { size: 10.5, color: C.offWhite, lineSpacingMultiple: 1.2 });
  });

  card(s, 6.85, 1.45, 6.0, 5.7, C.darkCard);
  addLabel(s, "🔜  In Progress / Next", 7.1, 1.6, 5.5, 0.38, { size: 13, bold: true, color: C.amber });
  next.forEach((n, i) => {
    addLabel(s, `• ${n}`, 7.1, 2.08 + i * 0.95, 5.5, 0.85, { size: 10.5, color: C.offWhite, lineSpacingMultiple: 1.2 });
  });

  pageNum(s, 9);
}

// ═══════════════════════════════════════════════════════════════
// SLIDE 10 — REFERENCES
// ═══════════════════════════════════════════════════════════════
{
  const s = slide();
  slideHeader(s, "Review 2 · Slide 10", "References");

  const refs = [
    "[1] Kirillov, A. et al. \"Segment Anything.\" IEEE/CVF ICCV, 2023.",
    "[2] Ma, J. et al. \"Segment Anything in Medical Images.\" Nature Communications, vol. 15, 654, 2024.",
    "[3] Butoi, V. I. et al. \"UniverSeg: Universal Medical Image Segmentation.\" IEEE/CVF ICCV, 2023.",
    "[4] Gao, Y. et al. \"Show and Segment: Universal Medical Image Segmentation via In-Context Learning.\" IEEE/CVF CVPR, 2025.",
    "[5] Antonelli, M. et al. \"The Medical Segmentation Decathlon.\" Nature Communications, vol. 13, 4128, 2022.",
    "[6] Chai et al. \"Onco-Seg: Promptable Concept Segmentation for Oncology.\" 2026.",
    "[7] Isensee, F. et al. \"nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.\" Nature Methods, vol. 18, 2021.",
  ];

  refs.forEach((r, i) => {
    addLabel(s, r, 0.6, 1.5 + i * 0.73, 12.1, 0.65, { size: 11.5, color: C.offWhite, lineSpacingMultiple: 1.2 });
  });

  pageNum(s, 10);
}

pres.writeFile({ fileName: path.join(__dirname, "Capstone_Review2.pptx") }).then(() => console.log("done"));