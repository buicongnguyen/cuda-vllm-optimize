const navToggle = document.querySelector("[data-nav-toggle]");
const navLinks = document.querySelector("[data-nav-links]");

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const open = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(open));
  });

  navLinks.addEventListener("click", (event) => {
    if (event.target instanceof HTMLAnchorElement) {
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

const reveals = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.08 },
  );
  reveals.forEach((element) => revealObserver.observe(element));
} else {
  reveals.forEach((element) => element.classList.add("visible"));
}

const ttftInput = document.querySelector("[data-ttft]");
const tpotInput = document.querySelector("[data-tpot]");
const scoreOutput = document.querySelector("[data-score]");
const scoreDetail = document.querySelector("[data-score-detail]");
const ttftContributionOutput = document.querySelector("[data-ttft-contribution]");
const tpotContributionOutput = document.querySelector("[data-tpot-contribution]");
const ttftBar = document.querySelector("[data-ttft-bar]");
const tpotBar = document.querySelector("[data-tpot-bar]");

function updateContribution(output, bar, value) {
  if (output instanceof HTMLElement) {
    output.textContent = value.toFixed(2);
  }
  if (bar instanceof HTMLElement) {
    const percentage = Math.max(0, Math.min(100, (value / 50) * 100));
    bar.style.width = `${percentage}%`;
    bar.parentElement?.setAttribute("aria-label", `${value.toFixed(2)} trên tối đa 50 điểm`);
  }
}

function calculateScore() {
  if (!(ttftInput instanceof HTMLInputElement) ||
      !(tpotInput instanceof HTMLInputElement) ||
      !(scoreOutput instanceof HTMLElement)) {
    return;
  }

  const ttft = Number(ttftInput.value);
  const tpot = Number(tpotInput.value);
  if (!Number.isFinite(ttft) || !Number.isFinite(tpot) || ttft < 0 || tpot < 0) {
    scoreOutput.textContent = "—";
    updateContribution(ttftContributionOutput, ttftBar, 0);
    updateContribution(tpotContributionOutput, tpotBar, 0);
    return;
  }

  const ttftPart = (400 - ttft) / 390;
  const tpotPart = (10 - tpot) / 9;
  const ttftContribution = 50 * ttftPart ** 2;
  const tpotContribution = 50 * tpotPart ** 2;
  const score = ttftContribution + tpotContribution;
  scoreOutput.textContent = score.toFixed(2);
  updateContribution(ttftContributionOutput, ttftBar, ttftContribution);
  updateContribution(tpotContributionOutput, tpotBar, tpotContribution);

  if (scoreDetail instanceof HTMLElement) {
    const target = 72;
    const state = score >= target ? "đạt" : `còn thiếu ${(target - score).toFixed(2)}`;
    scoreDetail.textContent = `Mốc 72 ERS: ${state}. Đây là công thức được trích dẫn, chưa phải evaluator chính thức.`;
  }
}

[ttftInput, tpotInput].forEach((input) => input?.addEventListener("input", calculateScore));
calculateScore();

const lessons = document.querySelectorAll(".lesson[id]");
const tocLinks = [...document.querySelectorAll(".toc a[href^='#']")];

if (lessons.length && tocLinks.length && "IntersectionObserver" in window) {
  const lessonObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      tocLinks.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
      });
    },
    { rootMargin: "-20% 0px -65%", threshold: [0.05, 0.3, 0.6] },
  );
  lessons.forEach((lesson) => lessonObserver.observe(lesson));
}

const markdownArticle = document.querySelector("[data-doc-source]");
const documentToc = document.querySelector("[data-doc-toc]");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInline(value) {
  let rendered = escapeHtml(value);
  rendered = rendered.replace(/`([^`]+)`/g, "<code>$1</code>");
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  rendered = rendered.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|[^\s):]+(?:#[^\s)]*)?)\)/g, '<a href="$2">$1</a>');
  return rendered;
}

function headingId(value, usedIds) {
  const root = value
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replaceAll("đ", "d")
    .replaceAll("Đ", "D")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "") || "section";
  let candidate = root;
  let suffix = 2;
  while (usedIds.has(candidate)) {
    candidate = `${root}-${suffix}`;
    suffix += 1;
  }
  usedIds.add(candidate);
  return candidate;
}

function tableCells(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isTableDivider(line) {
  const cells = tableCells(line);
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function startsBlock(lines, index) {
  const line = lines[index] ?? "";
  const next = lines[index + 1] ?? "";
  return /^#{1,3}\s+/.test(line) || /^```/.test(line) || /^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line) || (line.includes("|") && isTableDivider(next));
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  const usedIds = new Set();
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const title = heading[2].trim();
      if (level > 1) {
        const id = headingId(title, usedIds);
        output.push(`<h${level} id="${id}">${renderInline(title)}<a class="heading-anchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a></h${level}>`);
      }
      index += 1;
      continue;
    }

    if (/^```/.test(line)) {
      const language = line.slice(3).trim();
      const code = [];
      index += 1;
      while (index < lines.length && !/^```/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      index += 1;
      output.push(`<pre><code${language ? ` data-language="${escapeHtml(language)}"` : ""}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    if (line.includes("|") && isTableDivider(lines[index + 1] ?? "")) {
      const headers = tableCells(line);
      index += 2;
      const rows = [];
      while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
        rows.push(tableCells(lines[index]));
        index += 1;
      }
      output.push('<div class="prose-table"><table><thead><tr>');
      headers.forEach((cell) => output.push(`<th scope="col">${renderInline(cell)}</th>`));
      output.push("</tr></thead><tbody>");
      rows.forEach((row) => {
        output.push("<tr>");
        headers.forEach((_, cellIndex) => output.push(`<td>${renderInline(row[cellIndex] ?? "")}</td>`));
        output.push("</tr>");
      });
      output.push("</tbody></table></div>");
      continue;
    }

    const listMatch = line.match(/^([-*]|\d+\.)\s+(.+)$/);
    if (listMatch) {
      const ordered = /\d+\./.test(listMatch[1]);
      const tag = ordered ? "ol" : "ul";
      output.push(`<${tag}>`);
      while (index < lines.length) {
        const item = lines[index].match(ordered ? /^\d+\.\s+(.+)$/ : /^[-*]\s+(.+)$/);
        if (!item) break;
        const parts = [item[1].trim()];
        index += 1;
        while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
          parts.push(lines[index].trim());
          index += 1;
        }
        output.push(`<li>${renderInline(parts.join(" "))}</li>`);
        while (index < lines.length && !lines[index].trim()) index += 1;
      }
      output.push(`</${tag}>`);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim() && !startsBlock(lines, index)) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    output.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
  }

  return output.join("\n");
}

function buildDocumentToc(article, toc) {
  if (!(toc instanceof HTMLElement)) return;
  const headings = [...article.querySelectorAll("h2, h3")];
  toc.replaceChildren();
  headings.forEach((heading) => {
    const link = document.createElement("a");
    link.href = `#${heading.id}`;
    link.textContent = heading.firstChild?.textContent?.trim() || heading.textContent.trim();
    if (heading.tagName === "H3") link.classList.add("subsection-link");
    toc.append(link);
  });
}

if (markdownArticle instanceof HTMLElement) {
  const source = markdownArticle.dataset.docSource;
  if (source) {
    fetch(source)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then((markdown) => {
        markdownArticle.innerHTML = renderMarkdown(markdown);
        markdownArticle.removeAttribute("aria-live");
        buildDocumentToc(markdownArticle, documentToc);
      })
      .catch(() => {
        markdownArticle.innerHTML = `<h2>Không thể tải tài liệu</h2><p>Mở <a href="${escapeHtml(source)}">Markdown source</a> hoặc thử tải lại trang.</p>`;
      });
  }
}
