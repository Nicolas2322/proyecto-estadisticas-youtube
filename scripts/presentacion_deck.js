// Presentación del proyecto final: Estadísticas de YouTube (PLN)
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fs = require("fs");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const SPACE = process.env.SPACE_URL || "huggingface.co/spaces/yohan23320/estadisticas-youtube";
const MODELO = process.env.MODEL_URL || "huggingface.co/yohan23320/youtube-sentimiento-vistas";
const REPO = "github.com/Nicolas2322/proyecto-estadisticas-youtube";

const THEME = {
  name: "Estadisticas YouTube",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1E24", lt1: "FFFFFF", dk2: "2B2F38", lt2: "F2F3F5",
    accent1: "D7263D", accent2: "1F7A8C", accent3: "F2A541", accent4: "6C757D", accent5: "8FBCC8", accent6: "F6C9CF",
    hlink: "1F7A8C", folHlink: "6C757D",
  },
};
const H = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Proyecto final: Estadísticas de YouTube";
pres.author = "Yohan Nicolás Gómez Castañeda";
const C = pres.SchemeColor;

// ---------- layouts
pres.defineSlideMaster({
  title: "PORTADA", background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.0, w: 11.9, h: 1.9, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.05, w: 11.9, h: 0.9, fontSize: 20, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENIDO", background: { color: H.lt1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Estadísticas de YouTube · Procesamiento de Lenguaje Natural", options: { x: 0.6, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});
pres.defineSlideMaster({
  title: "OSCURO", background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 28, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- utilidades
async function icono(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function circuloIcono(slide, Comp, x, y, d, fondo, colorIcono, nombre) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fondo }, line: { type: "none" }, objectName: nombre + " círculo" });
  const p = d * 0.26;
  slide.addImage({ data: await icono(Comp, colorIcono), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: nombre + " ícono" });
}
const tx = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0, valign: "top", color: C.text1 }, o));
const sombra = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function tarjeta(s, x, y, w, h, nombre, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: color || H.lt2 }, line: { type: "none" }, objectName: nombre });
}
function flecha(s, x1, y1, x2, y2, color, nombre) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1, line: { color: color || H.accent4, width: 1.75, endArrowType: "triangle" }, objectName: nombre });
}
const chartTexto = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
  catAxisLabelColor: "4A4F59", valAxisLabelColor: "4A4F59", dataLabelColor: "1B1E24", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 12 };

(async () => {
  // ================================================================ 1. Portada
  pres.addSection({ title: "Inicio" });
  let s = pres.addSlide({ masterName: "PORTADA", sectionTitle: "Inicio" });
  s.addText("Estadísticas de YouTube: sentimiento de la audiencia y vistas estimadas a partir de los comentarios", { placeholder: "title" });
  s.addText("Proyecto final · Procesamiento de Lenguaje Natural", { placeholder: "body" });
  await circuloIcono(s, fa.FaYoutube, 0.7, 0.75, 0.95, H.accent1, H.lt1, "Logo");
  tx(s, [
    { text: "Yohan Nicolás Gómez Castañeda", options: { bold: true, breakLine: true } },
    { text: "Especialización en Inteligencia Artificial · Universidad de Cundinamarca", options: { breakLine: true } },
    { text: "Docente: Paul Alexander Díaz Montaña · 22 de noviembre de 2026" },
  ], { x: 0.7, y: 5.3, w: 6.5, h: 1.1, fontSize: 15, color: C.background1, paraSpaceAfter: 3 });
  tx(s, [{ text: "Servicio en la nube", options: { bold: true, breakLine: true, color: C.accent3 } }, { text: SPACE }],
    { x: 7.3, y: 5.3, w: 5.4, h: 1.1, fontSize: 13, color: C.background1, align: "right" });
  s.addNotes("Duración total prevista: 13 minutos. (0:40) Presento el proyecto final: un servicio de procesamiento de lenguaje natural que, a partir de los comentarios de un video de YouTube, clasifica el sentimiento de cada comentario y estima cuántas vistas tendrá el video, con un rango de confianza. El modelo está desplegado en Hugging Face y se puede usar desde una interfaz web o como API.");

  // ================================================================ 2. Problemática
  pres.addSection({ title: "Problema y solución" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Problema y solución" });
  s.addText("Los comentarios dicen mucho, pero nadie alcanza a leerlos", { placeholder: "title" });
  tarjeta(s, 0.6, 1.5, 4.0, 5.1, "Dato de contexto", H.dk1);
  tx(s, "500 h", { x: 0.9, y: 1.85, w: 3.5, h: 1.2, fontSize: 66, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  tx(s, "de video se suben a YouTube cada minuto (Statista, 2022)", { x: 0.9, y: 3.1, w: 3.4, h: 0.9, fontSize: 16, color: C.background1 });
  tx(s, "Un canal que publica decenas de videos al mes acumula miles de comentarios: texto corto, informal, con ironía y emojis.",
    { x: 0.9, y: 4.25, w: 3.4, h: 1.9, fontSize: 14, color: C.accent5 });
  const preg = [
    [fa.FaChartLine, "¿Cuánto alcance tendrá el video?", "Hoy se espera semanas a que las vistas se estabilicen, cuando ya es tarde para reasignar la pauta."],
    [fa.FaComments, "¿Qué opina la audiencia?", "Se leen comentarios a mano: no escala y cada persona los juzga distinto. Un léxico como VADER acierta poco (F1-macro 0,596)."],
    [fa.FaBalanceScale, "¿Cómo comparar campañas y categorías?", "Sin un indicador cuantitativo no se pueden comparar videos, periodos ni tipos de contenido."],
  ];
  for (let i = 0; i < preg.length; i++) {
    const y = 1.6 + i * 1.7;
    await circuloIcono(s, preg[i][0], 5.1, y, 0.85, H.accent6, H.accent1, "Pregunta " + (i + 1));
    tx(s, preg[i][1], { x: 6.2, y: y - 0.02, w: 6.4, h: 0.45, fontSize: 20, bold: true });
    tx(s, preg[i][2], { x: 6.2, y: y + 0.48, w: 6.4, h: 0.9, fontSize: 15, color: C.text2 });
  }
  s.addNotes("(1:20) La problemática: un equipo de mercadeo o de contenido publica videos sin poder anticipar su alcance y sin medir a escala qué opina la audiencia. Los comentarios concentran ambas señales, pero son texto no estructurado y su lectura manual no escala. En el análisis exploratorio vimos además que la categoría explica el 30 % de la variación de las vistas y que el sentimiento cambia mucho entre categorías: los videos de noticias reciben hasta un 55 % de comentarios negativos, frente a menos del 4 % en música.");

  // ================================================================ 3. Solución
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Problema y solución" });
  s.addText("La solución: de los comentarios al sentimiento y las vistas", { placeholder: "title" });
  const pasos = [[fa.FaKeyboard, "Entrada", "Comentarios del video (o su enlace), categoría y días desde la publicación"],
    [fa.FaBrain, "Etapa A", "RoBERTa clasifica cada comentario: negativo, neutral o positivo"],
    [fa.FaLayerGroup, "Etapa B", "Stacking de 4 modelos estima las vistas con un intervalo del 90 %"],
    [fa.FaClipboardCheck, "Salida", "Sentimiento por comentario, vistas estimadas y su rango, % de negativos"]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1;
    tarjeta(s, x, 1.45, 2.8, 2.05, "Paso " + (i + 1));
    await circuloIcono(s, pasos[i][0], x + 0.2, 1.62, 0.62, i === 0 || i === 3 ? H.accent1 : H.accent2, H.lt1, "Paso " + (i + 1));
    tx(s, pasos[i][1], { x: x + 0.95, y: 1.75, w: 1.75, h: 0.4, fontSize: 18, bold: true });
    tx(s, pasos[i][2], { x: x + 0.2, y: 2.4, w: 2.45, h: 1.05, fontSize: 13, color: C.text2 });
    if (i < 3) flecha(s, x + 2.82, 2.47, x + 3.08, 2.47, H.accent4, "Flecha paso " + (i + 1));
  }
  s.addImage({ path: "demo.png", x: 0.6, y: 3.8, w: 7.85, h: 7.85 * 475 / 1568, objectName: "Captura de la interfaz", altText: "Interfaz web del servicio con el sentimiento de cada comentario y las vistas estimadas" });
  tx(s, [{ text: "Ejemplo real del servicio", options: { bold: true, breakLine: true } },
    { text: "Cuatro comentarios de un tutorial de 30 días: dos positivos, una queja por el audio (negativo) y una pregunta neutral. Vistas estimadas: 806.905, con un rango del 90 % entre 145.220 y 4,48 millones." }],
    { x: 8.85, y: 3.85, w: 3.85, h: 2.6, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  s.addNotes("(1:00) La solución tiene dos etapas encadenadas. La etapa A clasifica cada comentario con un modelo de lenguaje preentrenado; la etapa B agrega esas predicciones por video junto con la interacción que generan los comentarios, sus temas, la antigüedad y la categoría, y estima las vistas. Lo importante para el usuario es que la estimación viene con un rango del 90 %: el modelo distingue órdenes de magnitud, no promete una cifra exacta.");

  // ================================================================ 4. Fuente de información
  pres.addSection({ title: "Datos y diseño" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y diseño" });
  s.addText("Fuente de información: pública para desarrollar, propia para operar", { placeholder: "title" });
  tarjeta(s, 0.6, 1.45, 6.0, 5.2, "Fuente de desarrollo");
  await circuloIcono(s, fa.FaDatabase, 0.85, 1.65, 0.7, H.accent2, H.lt1, "Kaggle");
  tx(s, "Desarrollo: YouTube Statistics (Kaggle)", { x: 1.75, y: 1.75, w: 4.7, h: 0.5, fontSize: 18, bold: true });
  const filas = [["Archivo", "Registros", "Contenido"],
    ["videos-stats.csv", "1.881 → 1.867", "Título, fecha, categoría (41), likes, comentarios, vistas"],
    ["comments.csv", "18.409 → 18.139", "Texto, likes y sentimiento etiquetado (0, 1, 2)"]];
  s.addTable(filas.map((r, i) => r.map(c => ({ text: c, options: { bold: i === 0, color: i === 0 ? H.lt1 : H.dk1, fill: { color: i === 0 ? H.dk2 : H.lt1 } } }))),
    { x: 0.85, y: 2.55, w: 5.5, colW: [1.55, 1.45, 2.5], fontSize: 12, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D5D8DD" }, valign: "middle", objectName: "Tabla fuente" });
  tx(s, [
    { text: "Calidad: ", options: { bold: true } }, { text: "12 videos y 249 comentarios repetidos, contadores ocultos (−1), clases desbalanceadas (61,9 % positivo · 25,3 % neutral · 12,8 % negativo).", options: { breakLine: true } },
    { text: "Trazabilidad: ", options: { bold: true } }, { text: "registro de limpieza con conteos antes y después, huellas SHA-256 y respaldo de los 284 registros retirados.", options: { breakLine: true } },
    { text: "Acceso: ", options: { bold: true } }, { text: "4,3 MB en Google Drive, leídos desde Colab." },
  ], { x: 0.85, y: 4.15, w: 5.5, h: 2.4, fontSize: 13, color: C.text2, paraSpaceAfter: 6 });
  tarjeta(s, 6.9, 1.45, 5.85, 5.2, "Fuente de operación", H.dk1);
  await circuloIcono(s, fa.FaYoutube, 7.15, 1.65, 0.7, H.accent1, H.lt1, "API YouTube");
  tx(s, "Operación: el canal propio de la empresa", { x: 8.05, y: 1.75, w: 4.55, h: 0.5, fontSize: 18, bold: true, color: C.background1 });
  tx(s, [
    { text: "API de datos de YouTube v3: ", options: { bold: true, color: C.accent3 } }, { text: "commentThreads.list (comentarios) y videos.list (fecha y vistas), con cuota diaria gratuita.", options: { breakLine: true } },
    { text: "Ventajas: ", options: { bold: true, color: C.accent3 } }, { text: "datos reales del negocio, mediciones del mismo video en el tiempo y una muestra propia para recalibrar.", options: { breakLine: true } },
    { text: "Ley 1581 de 2012 (Habeas Data): ", options: { bold: true, color: C.accent3 } }, { text: "el servicio no lee ni guarda el nombre del autor ni el texto de las consultas; solo métricas agregadas.", options: { breakLine: true } },
    { text: "Cautela: ", options: { bold: true, color: C.accent3 } }, { text: "un modelo independiente coincide en 89 % con las etiquetas de Kaggle; pudieron generarse de forma automática." },
  ], { x: 7.15, y: 2.6, w: 5.35, h: 3.9, fontSize: 14, color: C.background1, paraSpaceAfter: 8 });
  s.addNotes("(1:20) Para desarrollar usamos el conjunto YouTube Statistics de Kaggle: 1.881 videos y 18.409 comentarios etiquetados. Tras la limpieza quedan 1.867 videos y 18.139 comentarios. La limpieza se documentó con un registro reversible, como pidió el profesor: duplicados exactos frente a lógicos, contadores ocultos tratados como faltantes y respaldo de todo lo retirado. En operación, la fuente sería el canal propio de la empresa a través de la API de YouTube, cumpliendo la Ley 1581: no guardamos nombres de autores ni el texto de las consultas.");

  // ================================================================ 5. Diseño esquemático
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y diseño" });
  s.addText("Diseño esquemático: dónde vive cada componente", { placeholder: "title" });
  const zona = (x, y, w, h, t, nombre, color) => { tarjeta(s, x, y, w, h, nombre, color); tx(s, t, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.35, fontSize: 13, bold: true, color: C.accent4 }); };
  zona(0.6, 1.4, 3.7, 5.3, "ENTRENAMIENTO", "Zona entrenamiento");
  zona(4.7, 1.4, 4.25, 5.3, "NUBE · HUGGING FACE", "Zona nube", "E8F1F3");
  zona(9.35, 1.4, 3.4, 5.3, "USUARIOS", "Zona usuarios");
  const comp = async (x, y, w, Ic, t, d, col, nombre) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.05, rectRadius: 0.06, fill: { color: H.lt1 }, line: { color: "D5D8DD", width: 0.75 }, shadow: sombra(), objectName: nombre });
    await circuloIcono(s, Ic, x + 0.12, y + 0.2, 0.62, col, H.lt1, nombre);
    tx(s, t, { x: x + 0.85, y: y + 0.12, w: w - 0.95, h: 0.38, fontSize: 14, bold: true });
    tx(s, d, { x: x + 0.85, y: y + 0.5, w: w - 0.95, h: 0.5, fontSize: 11, color: C.text2 });
  };
  await comp(0.8, 2.0, 3.3, fa.FaGoogleDrive, "Google Drive", "CSV de Kaggle y predicciones fuera de fold", H.accent3, "Drive");
  await comp(0.8, 3.55, 3.3, fa.FaPython, "Google Colab (GPU T4)", "Limpieza, ajuste de RoBERTa, stacking y validación", H.accent2, "Colab");
  await comp(0.8, 5.1, 3.3, fa.FaGithub, "GitHub", "Código, notebooks, resultados y versiones", H.dk2, "GitHub");
  await comp(4.9, 2.0, 3.85, fa.FaBoxOpen, "Repositorio de modelos", "RoBERTa ajustado + etapa_b.joblib (stacking, tópicos, q del intervalo)", H.accent2, "Hub");
  await comp(4.9, 3.55, 3.85, fa.FaServer, "Space Gradio (ZeroGPU)", "Interfaz web, API /predecir, monitoreo de latencia y deriva", H.accent1, "Space");
  await comp(4.9, 5.1, 3.85, fa.FaYoutube, "API de datos de YouTube", "Comentarios de un video por su enlace (opcional)", H.accent1, "YouTube API");
  await comp(9.55, 2.0, 3.0, fa.FaDesktop, "Analista", "Navegador web: interfaz con pestañas", H.accent4, "Analista");
  await comp(9.55, 3.55, 3.0, fa.FaCode, "Otros sistemas", "Cliente Python o HTTP (curl)", H.accent4, "Sistemas");
  await comp(9.55, 5.1, 3.0, fa.FaChartBar, "Monitoreo", "Registro JSON sin texto, alerta de deriva", H.accent4, "Monitoreo");
  flecha(s, 2.45, 3.05, 2.45, 3.53, H.accent4, "Drive a Colab");
  flecha(s, 2.45, 4.6, 2.45, 5.08, H.accent4, "Colab a GitHub");
  flecha(s, 4.1, 3.9, 4.88, 2.6, H.accent2, "Colab publica modelos");
  flecha(s, 6.8, 3.05, 6.8, 3.53, H.accent2, "Space descarga modelos");
  flecha(s, 6.8, 5.08, 6.8, 4.62, H.accent1, "YouTube a Space");
  flecha(s, 8.75, 3.95, 9.53, 2.55, H.accent1, "Space a analista");
  flecha(s, 8.75, 4.05, 9.53, 4.05, H.accent1, "Space a sistemas");
  flecha(s, 8.75, 4.2, 9.53, 5.55, H.accent1, "Space a monitoreo");
  s.addNotes("(1:30) Este es el diseño esquemático. A la izquierda, el entrenamiento: los datos viven en Google Drive y el notebook de Colab, con GPU T4, entrena y valida todo; el código y los resultados se versionan en GitHub. En el centro, la nube: el notebook publica los modelos en un repositorio de Hugging Face, y el Space con Gradio los descarga al arrancar. Así, una nueva versión del modelo se publica sin tocar el código y se puede volver a la anterior. A la derecha, quién lo usa: un analista desde el navegador, otros sistemas por API y el tablero de monitoreo.");

  // ================================================================ 6. Pipeline
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Datos y diseño" });
  s.addText("Pipeline: ocho etapas con retornos previstos", { placeholder: "title" });
  const et = [
    ["1", "Adquisición", "Kaggle o API de YouTube; huella SHA-256"],
    ["2", "Limpieza", "Duplicados, −1 como faltante, registro reversible"],
    ["3", "Preprocesamiento", "@user/http, tokens BPE; TF-IDF con n-gramas"],
    ["4", "Características", "Sentimiento, VADER, likes y Gini, 20 tópicos, edad, categoría"],
    ["5", "Modelado", "RoBERTa ajustado; stacking XGB + HGB + RF + Ridge"],
    ["6", "Evaluación", "5 folds, CV anidada, McNemar, Wilcoxon, conformal"],
    ["7", "Despliegue", "Modelos versionados, Gradio y API en Hugging Face"],
    ["8", "Monitoreo", "Latencia, deriva de clases y entradas, reentreno"],
  ];
  for (let i = 0; i < 8; i++) {
    const fila = Math.floor(i / 4), col = fila === 0 ? i : 7 - i;
    const x = 0.6 + col * 3.1, y = fila === 0 ? 1.5 : 4.35;
    const op = i >= 6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 2.75, h: 2.05, rectRadius: 0.08, fill: { color: op ? "FBE4E7" : H.lt2 }, line: { type: "none" }, objectName: "Etapa " + et[i][0] });
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, fill: { color: op ? H.accent1 : H.accent2 }, line: { type: "none" }, objectName: "Número etapa " + et[i][0] });
    tx(s, et[i][0], { x: x + 0.2, y: y + 0.2, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    tx(s, et[i][1], { x: x + 0.88, y: y + 0.27, w: 1.8, h: 0.45, fontSize: 17, bold: true });
    tx(s, et[i][2], { x: x + 0.2, y: y + 0.92, w: 2.4, h: 1.05, fontSize: 13, color: C.text2 });
  }
  for (let c = 0; c < 3; c++) {
    flecha(s, 0.6 + c * 3.1 + 2.77, 2.52, 0.6 + (c + 1) * 3.1 - 0.02, 2.52, H.accent2, "Flecha superior " + c);
    flecha(s, 0.6 + (c + 1) * 3.1 - 0.02, 5.37, 0.6 + c * 3.1 + 2.77, 5.37, c === 2 ? H.accent2 : H.accent1, "Flecha inferior " + c);
  }
  flecha(s, 10.9, 3.57, 10.9, 4.33, H.accent2, "Características a modelado");
  s.addShape(pres.shapes.LINE, { x: 1.97, y: 3.6, w: 0.001, h: 0.72, line: { color: H.accent1, width: 1.5, dashType: "dash", beginArrowType: "triangle" }, objectName: "Retorno monitoreo" });
  tx(s, "deriva → datos nuevos", { x: 2.1, y: 3.78, w: 2.3, h: 0.35, fontSize: 11, italic: true, color: C.accent1 });
  s.addShape(pres.shapes.LINE, { x: 8.17, y: 3.6, w: 0.001, h: 0.72, line: { color: H.accent1, width: 1.5, dashType: "dash", beginArrowType: "triangle" }, objectName: "Retorno evaluación" });
  tx(s, "métrica bajo meta → volver", { x: 5.6, y: 3.78, w: 2.45, h: 0.35, fontSize: 11, italic: true, color: C.accent1, align: "right" });
  s.addNotes("(1:40) El pipeline sigue las fases vistas en clase: adquisición, preprocesamiento, extracción de características, modelado, evaluación, despliegue y operación. Le agregamos una etapa explícita de limpieza con registro y otra de monitoreo. No es lineal: las líneas punteadas son los retornos previstos. Si una métrica queda bajo la meta se vuelve a las características, como pasó cuando cambiamos la SVM por RoBERTa; si el monitoreo detecta deriva, se vuelve a la adquisición con datos nuevos. Dos reglas evitan resultados optimistas: los datos de prueba nunca participan en ningún ajuste, y las variables de cada video se construyen con predicciones fuera de fold.");

  // ================================================================ 7. Modelos y selección
  pres.addSection({ title: "Resultados" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText("69 modelos comparados: el salto vino de la representación", { placeholder: "title" });
  const nomA = ["TextBlob (léxico)", "VADER (léxico)", "Naive Bayes compl.", "Regresión logística", "SVM TF-IDF (inicial)", "SVM híbrido + GloVe", "RoBERTa sin ajuste", "RoBERTa ajustado"];
  const valA = [0.557, 0.596, 0.624, 0.685, 0.710, 0.757, 0.855, 0.901];
  s.addChart(pres.charts.BAR, [{ name: "F1-macro", labels: nomA, values: valA }], Object.assign({
    x: 0.6, y: 1.4, w: 7.4, h: 5.35, barDir: "bar", showTitle: true, title: "Etapa A · F1-macro (5 folds)", titleFontSize: 15, titleColor: "1B1E24",
    chartColors: ["B7C4CB", "B7C4CB", "B7C4CB", "B7C4CB", "F2A541", "8FBCC8", "8FBCC8", "D7263D"], invertedColors: undefined,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.4, valAxisMaxVal: 1.0, valAxisMajorUnit: 0.1,
    valGridLine: { color: "E3E5E8", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45,
  }, chartTexto));
  tx(s, [{ text: "Cuántos modelos", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "17 clasificadores de sentimiento, 33 modelos de vistas y 19 de éxito relativo, más 74 combinaciones de hiperparámetros y 125 ensayos de Optuna.", options: { breakLine: true } }],
    { x: 8.4, y: 1.5, w: 4.3, h: 1.7, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  tx(s, [{ text: "Cómo se eligió", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Mismas 5 particiones (semilla 42) para todos, CV anidada, MLflow (25 iteraciones), McNemar y Wilcoxon, SHAP y ablación. Un cambio solo se adopta si mejora más que la variación entre folds.", options: { breakLine: true } }],
    { x: 8.4, y: 3.3, w: 4.3, h: 1.9, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  tx(s, [{ text: "Más allá de la precisión", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "Modelo abierto en vez de un LLM por API: sin costo por consulta y sin enviar comentarios a terceros." }],
    { x: 8.4, y: 5.3, w: 4.3, h: 1.4, fontSize: 14, color: C.text2, paraSpaceAfter: 4 });
  s.addNotes("(1:20) Probamos 69 configuraciones, todas con las mismas cinco particiones para que fueran comparables. En sentimiento, seis iteraciones de ingeniería de variables sobre la SVM se estancaron en 0,757. El salto llegó al cambiar la representación: RoBERTa entrenado con tuits, incluso sin ajuste, ya daba 0,855, y ajustado con nuestras etiquetas llegó a 0,901. Como dijo el profesor, no hay que reinventar la rueda: ajustamos un modelo ya entrenado. Y no se eligió solo por precisión: un modelo abierto no cobra por consulta ni envía comentarios a terceros.");

  // ================================================================ 8. Resultados
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText("Resultados: todas las metas técnicas se cumplieron", { placeholder: "title" });
  const kpi = [["0,901", "F1-macro de sentimiento", "inicial 0,710 · meta ≥ 0,85", H.accent1],
    ["0,880", "F1 de la clase negativa", "inicial 0,571 · meta ≥ 0,80", H.accent1],
    ["0,791", "R² sobre log10(vistas)", "inicial 0,788 · meta ≥ 0,78", H.accent2],
    ["0,901", "Cobertura del intervalo del 90 %", "meta entre 0,88 y 0,92", H.accent2]];
  for (let i = 0; i < 4; i++) {
    const x = 0.6 + i * 3.1;
    tarjeta(s, x, 1.45, 2.8, 2.35, "Indicador " + (i + 1));
    tx(s, kpi[i][0], { x: x + 0.2, y: 1.6, w: 2.4, h: 0.95, fontSize: 48, bold: true, color: kpi[i][3] === H.accent1 ? C.accent1 : C.accent2, fontFace: THEME.headFontFace });
    tx(s, kpi[i][1], { x: x + 0.2, y: 2.6, w: 2.4, h: 0.6, fontSize: 15, bold: true });
    tx(s, kpi[i][2], { x: x + 0.2, y: 3.25, w: 2.4, h: 0.45, fontSize: 12, color: C.accent4 });
  }
  const t2 = [["Métrica", "Diseño inicial", "Solución final"],
    ["Exactitud del sentimiento", "0,784", "0,927"],
    ["Comentarios negativos detectados (sensibilidad)", "—", "92,2 %"],
    ["Negativos confundidos con positivos", "—", "1,4 %"],
    ["Error típico de las vistas (factor)", "×2,28", "×2,23"],
    ["Éxito relativo en su categoría (ROC-AUC)", "0,863", "0,879"]];
  s.addTable(t2.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0, color: i === 0 ? H.lt1 : H.dk1, fill: { color: i === 0 ? H.dk2 : (i % 2 ? H.lt1 : H.lt2) }, align: j ? "center" : "left" } }))),
    { x: 0.6, y: 4.15, w: 7.6, colW: [4.4, 1.6, 1.6], fontSize: 13, fontFace: "Calibri", border: { type: "none" }, rowH: 0.42, valign: "middle", objectName: "Tabla resultados" });
  tarjeta(s, 8.6, 4.15, 4.15, 2.55, "Lectura de resultados", H.dk1);
  tx(s, [{ text: "Qué significa", options: { bold: true, color: C.accent3, breakLine: true } },
    { text: "Los errores de sentimiento bajaron un 66 %. Las vistas se estiman en orden de magnitud: en 9 de cada 10 videos el valor real cae dentro del rango que muestra el servicio." }],
    { x: 8.85, y: 4.35, w: 3.7, h: 2.2, fontSize: 14, color: C.background1, paraSpaceAfter: 4 });
  s.addNotes("(1:20) Los resultados, todos con validación cruzada de cinco folds en Colab con GPU. El sentimiento pasó de 0,710 a 0,901 de F1-macro; la clase negativa, la más importante para el negocio, de 0,571 a 0,880, y casi nunca se confunde un comentario negativo con uno positivo. Las vistas se estiman con R² de 0,791 y un error típico de un factor de 2,2; por eso entregamos el intervalo del 90 %, cuya cobertura real es 0,901.");

  // ================================================================ 9. Hallazgo
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Resultados" });
  s.addText("Hallazgo: el alcance depende de la interacción, no del tono", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [{ name: "R²", labels: ["Solo metadatos", "Sin likes de comentarios", "Modelo completo", "Sin variables de sentimiento"], values: [0.654, 0.674, 0.791, 0.793] }], Object.assign({
    x: 0.6, y: 1.4, w: 7.2, h: 5.3, barDir: "col", showTitle: true, title: "R² sobre log10(vistas) al retirar grupos de variables", titleFontSize: 15, titleColor: "1B1E24",
    chartColors: ["B7C4CB", "F2A541", "1F7A8C", "8FBCC8"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    valAxisMinVal: 0.5, valAxisMaxVal: 0.85, valAxisMajorUnit: 0.05, valGridLine: { color: "E3E5E8", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 60,
  }, chartTexto));
  const hall = [[fa.FaHeart, "Quitar el sentimiento no cambia nada", "R² 0,791 → 0,793: el tono de la conversación casi no predice las vistas (correlación parcial 0,05)."],
    [fa.FaThumbsUp, "Quitar los likes de los comentarios sí", "R² cae a 0,674: cuánta interacción generan los comentarios es la señal dominante."],
    [fa.FaLightbulb, "Qué hacer con esto", "El sentimiento se entrega como producto propio. Para estimar mejor las vistas se necesitan más comentarios y mediciones en el tiempo, no otro algoritmo."]];
  for (let i = 0; i < 3; i++) {
    const y = 1.55 + i * 1.75;
    await circuloIcono(s, hall[i][0], 8.2, y, 0.7, i === 2 ? H.accent3 : H.accent6, i === 2 ? H.dk1 : H.accent1, "Hallazgo " + (i + 1));
    tx(s, hall[i][1], { x: 9.1, y: y - 0.02, w: 3.65, h: 0.45, fontSize: 16, bold: true });
    tx(s, hall[i][2], { x: 9.1, y: y + 0.45, w: 3.65, h: 1.2, fontSize: 13, color: C.text2 });
  }
  s.addNotes("(1:00) El hallazgo más interesante sale del análisis de ablación. Si quitamos las siete variables de sentimiento, el R² no cambia. Si quitamos los likes de los comentarios, cae a 0,674. Es decir, para el alcance importa cuánta interacción generan los comentarios, no su tono. Por eso el sentimiento se entrega como producto en sí, útil para entender a la audiencia, y la mejora futura de las vistas pasa por más datos por video.");

  // ================================================================ 10. Problemas y soluciones
  pres.addSection({ title: "Proceso y despliegue" });
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Proceso y despliegue" });
  s.addText("Problemas encontrados y cómo se resolvieron", { placeholder: "title" });
  const prob = [["Problema", "Solución", "Resultado"],
    ["Contadores con −1 rompían el logaritmo", "Interpretarlos como contador oculto (faltante) e imputar", "Se conservan los videos, sin sesgo"],
    ["Duplicados que no eran copias exactas", "Separar duplicados exactos y lógicos; registro reversible", "284 registros con respaldo en CSV"],
    ["R² de 0,944 demasiado bueno", "Detectar la fuga (likes del video, ρ = 0,95) y excluirlos", "R² honesto de 0,79"],
    ["Clase negativa ignorada (12,8 %)", "F1-macro, pesos por clase y umbrales anidados", "F1 negativo 0,571 → 0,880"],
    ["Modelos clásicos estancados en 0,757", "Ajustar RoBERTa-Twitter en GPU de Colab", "F1-macro 0,901"],
    ["Error típico de ×2,2 oculto", "Intervalo conformal calibrado fuera de fold", "Cobertura real 0,901"],
    ["Código de entrenamiento y de uso distintos", "Un solo módulo (final.py) para evaluar y predecir", "Misma transformación en la demo"],
    ["Hugging Face cobra la CPU desde julio de 2026", "Space en ZeroGPU (gratis) con respaldo en CPU", "Servicio permanente sin costo"]];
  s.addTable(prob.map((r, i) => r.map((c, j) => ({ text: c, options: { bold: i === 0 || j === 0, color: i === 0 ? H.lt1 : H.dk1, fill: { color: i === 0 ? H.dk1 : (i % 2 ? H.lt1 : H.lt2) } } }))),
    { x: 0.6, y: 1.4, w: 12.15, colW: [3.85, 4.9, 3.4], fontSize: 13, fontFace: "Calibri", border: { type: "none" }, rowH: 0.56, valign: "middle", objectName: "Tabla problemas" });
  s.addNotes("(1:30) Los problemas más costosos no fueron de algoritmos sino de datos, de evaluación honesta y de despliegue. Destaco tres. Primero, un R² de 0,944 que parecía excelente era fuga de información: los likes del video y las vistas son casi la misma señal; los excluimos. Segundo, la clase negativa era ignorada por el desbalance; la resolvimos con F1-macro, pesos por clase y el cambio a RoBERTa. Tercero, en julio de 2026 Hugging Face dejó de ofrecer CPU gratis para Spaces de Gradio; migramos el servicio a ZeroGPU, que sigue siendo gratuito, con respaldo automático en CPU.");

  // ================================================================ 11. Despliegue
  s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: "Proceso y despliegue" });
  s.addText("Despliegue: el modelo en la nube, por interfaz y por API", { placeholder: "title" });
  const capt = fs.existsSync("space.png") ? "space.png" : "demo.png";
  const [cw, ch] = process.env.CAPT_SIZE ? process.env.CAPT_SIZE.split("x").map(Number) : [1568, 475];
  const anchoC = 7.4, altoC = Math.min(anchoC * ch / cw, 4.35);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 1.4, w: anchoC, h: altoC, fill: { color: H.lt1 }, line: { color: "D5D8DD", width: 0.75 }, shadow: sombra(), objectName: "Marco captura" });
  s.addImage({ path: capt, x: 0.6, y: 1.4, w: altoC * cw / ch, h: altoC, objectName: "Captura del Space", altText: "Servicio desplegado en Hugging Face Spaces" });
  tx(s, [{ text: "Interfaz: ", options: { bold: true } }, { text: SPACE }], { x: 0.6, y: 1.5 + altoC, w: 7.4, h: 0.35, fontSize: 13, color: C.accent2 });
  tarjeta(s, 0.6, 6.0, 7.4, 0.75, "Pestañas", H.lt2);
  tx(s, "Respuesta en 2 a 3 s · Pestañas: Predicción, Video, Monitoreo, API, Acerca", { x: 0.8, y: 6.17, w: 7.0, h: 0.4, fontSize: 14, bold: true, color: C.text2 });
  tarjeta(s, 8.35, 1.4, 4.4, 3.05, "Código API", H.dk1);
  tx(s, "API en Python", { x: 8.6, y: 1.55, w: 4.0, h: 0.4, fontSize: 16, bold: true, color: C.accent3 });
  tx(s, 'from gradio_client import Client\ncli = Client("yohan23320/\n      estadisticas-youtube")\nr = cli.predict(\n  "Great video!\\nbad audio",\n  "tutorial", 30,\n  api_name="/predecir")\nr["vistas_estimadas"]',
    { x: 8.6, y: 2.0, w: 4.0, h: 2.35, fontSize: 12, fontFace: "Courier New", color: C.background1 });
  const dep = [[fa.FaBolt, "ZeroGPU con respaldo en CPU", "Usa la GPU compartida si hay cuota; si no, la CPU con el mismo modelo"],
    [fa.FaHistory, "Versiones y rollback", "Modelos en un repositorio aparte; se vuelve a la versión anterior desde el historial"],
    [fa.FaHeartbeat, "Monitoreo", "Latencia, distribución de clases y longitud frente al entrenamiento; alerta de deriva"]];
  for (let i = 0; i < 3; i++) {
    const y = 4.65 + i * 0.72;
    await circuloIcono(s, dep[i][0], 8.35, y, 0.55, H.accent6, H.accent1, "Despliegue " + (i + 1));
    tx(s, [{ text: dep[i][1] + ": ", options: { bold: true } }, { text: dep[i][2] }], { x: 9.05, y: y - 0.02, w: 3.7, h: 0.68, fontSize: 12, color: C.text2 });
  }
  s.addNotes("(1:40) Aquí está el servicio desplegado en Hugging Face. Ojo: si nadie lo ha usado en un rato, la primera consulta tarda cerca de un minuto mientras el Space arranca; abrirlo antes de la presentación. Después responde en 2 a 3 segundos. Abro el enlace y muestro la demo en vivo: pego comentarios, elijo la categoría, y en segundos aparece el sentimiento de cada comentario y las vistas estimadas con su rango. La misma función se consume como API: con el cliente de Gradio en Python, o por HTTP con curl. La pestaña de monitoreo registra cada consulta sin guardar el texto y compara la distribución de las clases con la del entrenamiento para alertar si hay deriva. Los modelos viven en un repositorio aparte, versionado, así que se puede volver a una versión anterior.");

  // ================================================================ 12. Cierre
  pres.addSection({ title: "Cierre" });
  s = pres.addSlide({ masterName: "OSCURO", sectionTitle: "Cierre" });
  s.addText("Conclusiones y siguientes pasos", { placeholder: "title" });
  const concl = [["Problema resuelto", "Un analista obtiene en segundos el sentimiento de los comentarios y las vistas esperadas de un video, con su margen de error."],
    ["Diseño que se mantiene", "Dos etapas, módulo único para evaluar y servir, modelos versionados aparte del código y pipeline con retornos."],
    ["Lo que enseñan los datos", "El tono casi no explica el alcance; la interacción sí. El valor del sentimiento está en entender a la audiencia."]];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.5, w: 3.8, h: 2.6, rectRadius: 0.08, fill: { color: H.dk2 }, line: { type: "none" }, objectName: "Conclusión " + (i + 1) });
    tx(s, concl[i][0], { x: x + 0.25, y: 1.7, w: 3.3, h: 0.5, fontSize: 19, bold: true, color: C.accent3 });
    tx(s, concl[i][1], { x: x + 0.25, y: 2.3, w: 3.3, h: 1.7, fontSize: 15, color: C.background1 });
  }
  tx(s, "Siguientes pasos", { x: 0.6, y: 4.45, w: 6, h: 0.45, fontSize: 19, bold: true, color: C.accent5 });
  tx(s, [{ text: "Ampliar a cientos de comentarios por video y a mediciones repetidas con la API de YouTube", options: { bullet: true, breakLine: true } },
    { text: "Auditar una muestra de etiquetas a mano (kappa de Cohen) y recalibrar con datos del canal", options: { bullet: true, breakLine: true } },
    { text: "Modelo multilingüe para comentarios en español y otros idiomas", options: { bullet: true } }],
    { x: 0.6, y: 4.95, w: 7.6, h: 1.7, fontSize: 15, color: C.background1, paraSpaceAfter: 6 });
  tx(s, [{ text: "Servicio", options: { bold: true, color: C.accent3, breakLine: true } }, { text: SPACE, options: { breakLine: true } },
    { text: "Modelos", options: { bold: true, color: C.accent3, breakLine: true } }, { text: MODELO, options: { breakLine: true } },
    { text: "Código", options: { bold: true, color: C.accent3, breakLine: true } }, { text: REPO }],
    { x: 8.6, y: 4.45, w: 4.15, h: 2.3, fontSize: 12, color: C.background1, paraSpaceAfter: 2 });
  s.addNotes("(0:50) Para cerrar: el problema quedó resuelto con un servicio que cualquier analista usa desde el navegador o que otro sistema consume por API. El diseño es mantenible: los modelos están versionados aparte del código. Y los datos nos enseñaron que el tono casi no explica el alcance. Los siguientes pasos son más comentarios por video y mediciones en el tiempo, una auditoría manual de etiquetas y un modelo multilingüe. Gracias.");

  await pres.writeFile({ fileName: "Presentacion_Proyecto_Final_Gomez_Castaneda.pptx" });
  await applyTheme("Presentacion_Proyecto_Final_Gomez_Castaneda.pptx", THEME);
  console.log("ok");
})();
