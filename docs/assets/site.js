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
