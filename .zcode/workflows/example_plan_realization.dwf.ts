/* zcode-workflow
description: "Конвейер plan-m1: proposal → apply → verify → archive"
*/
// Конвейер OpenSpec для вехи M1 «Каркас» (docs/plan-m1-foundation.md):
// для каждого change'а строго по порядку: proposal → apply по группам задач со scoped-воротами
// → make install/lint/test → независимая проверка → архивация после PASS.
// Модели назначены пользователем: apply/proposal — GLM-5.3-Flash$high, verify — GLM-5.3$max.

const FLASH = "account:zai-individual-coding-plan/GLM-5.3-Flash$high";
const MAX = "account:zai-individual-coding-plan/GLM-5.3$max";

interface ChangeDef {
  /** Имя change'а (kebab-case). */
  name: string;
  /** Номер строки в таблице §4 плана. */
  row: number;
  /** Содержание из таблицы §4. */
  scope: string;
  /** Способ проверки из таблицы §4. */
  check: string;
}

const CHANGES: ChangeDef[] = [
  {
    name: "m1-tooling",
    row: 1,
    scope: "pyproject (deps: numpy, pandas, pydantic>=2, PyYAML; extras [dev]: pytest, ruff; конфиг ruff и pytest), Makefile (install/test/lint/fmt), пакеты-скелеты common/, storage/, simulator/, tests/, проверка .gitignore",
    check: "make install; import common, storage, simulator; make lint на пустых пакетах",
  },
  {
    name: "m1-common",
    row: 2,
    scope: "common/timeutil.py: to_iso/from_iso («YYYY-MM-DD HH:MM:SS», секунды всегда 00), add_minutes, bucket5; common/logging_setup.py",
    check: "тесты test_timeutil.py: roundtrip ISO; бакеты …:00–:04 → :00, :05–:09 → :05; переход часа/полуночи",
  },
  {
    name: "m1-config",
    row: 3,
    scope: "pydantic-модели секций конфига по скелету §5 плана, дефолты у каждой секции, extra=forbid, загрузка config.yaml, эталонный config.yaml в репо",
    check: "тесты test_config.py: конфиг целиком из дефолтов; эталонный файл валиден; неизвестный ключ — понятная ошибка с путём; seed читается",
  },
  {
    name: "m1-storage",
    row: 4,
    scope: "схема §6 плана: trips, zone_metrics_5min, forecasts + индексы, идемпотентный init; db.py (прагмы foreign_keys/WAL, транзакции); repo.py (батч-вставка, выборки); csv_export.py; демо python -m storage.demo",
    check: "тесты test_storage.py: повторный init без ошибок; roundtrip всех трёх таблиц; время — ISO-текст; CSV содержит те же строки; демо-скрипт",
  },
];

interface ProposalResult {
  /** Имя change'а. */
  change: string;
  /** Созданные артефакты (пути). */
  artifacts: string[];
  /** 2–3 предложения о содержании. */
  summary: string;
}

interface ApplyStep {
  /** Что сделано в группе, одним предложением. */
  summary: string;
  /** Тест-файлы этой группы (для scoped-ворот); [] если тестов в группе нет. */
  testFiles: string[];
  /** Есть ли в tasks.md ещё группы после этой. */
  moreGroups: boolean;
  /** Отклонения от design.md/плана, если были, иначе "". */
  deviations: string;
}

interface VerifyResult {
  /** Итог независимой проверки. */
  verdict: "PASS" | "FAIL";
  /** Критичные замечания, блокирующие архивацию. */
  critical: string[];
  /** Путь к отчёту в .opencode/queue/verify/. */
  reportPath: string;
}

interface ChangeOutcome {
  /** Имя change'а. */
  change: string;
  /** Итог этапа. */
  status: string;
  /** Заметки. */
  notes: string;
}

interface Finding {
  /** Где: change или файл. */
  where: string;
  /** Что произошло. */
  what: string;
  /** Доказательство. */
  evidence: string;
  /** Подтверждено независимой проверкой или детерминированной командой. */
  status: "verified" | "unconfirmed";
  /** Важность. */
  severity: "low" | "medium" | "high";
}

interface WorkflowReport {
  /** Ответ на запрос пользователя. */
  conclusion: string;
  findings: Finding[];
  /** Что прогонено и как. */
  verified: string[];
  /** Что не сделано и почему. */
  notCovered: string[];
}

artifact.board("conveyor", {
  title: "Конвейер изменений вехи M1",
  key: "change",
  status: "status",
  columns: ["proposal готов", "apply готов", "verify PASS", "архивирован", "ПРОВАЛЕН"],
  cardTitle: "change",
  detail: [{ field: "notes", label: "Заметки" }],
});

const PROPOSE_SYSTEM = `Ты — автор planning-артефактов OpenSpec в проекте skvoznoi_project (веха M1 «Каркас»).
1. Прочитай файл .zcode/skills/openspec-propose/SKILL.md и следуй ему.
2. Создавай ТОЛЬКО артефакты change'а в openspec/changes/<имя>/ (proposal, specs-delta, design, tasks). Код проекта и тесты не пиши.
3. Работай через openspec CLI: openspec new change <имя>, openspec status --change <имя> --json, openspec instructions <artifact> --change <имя> --json; в конце добейся зелёного openspec validate <имя> --strict (прогони сам).
4. Источник содержания — docs/plan-m1-foundation.md; правила из контекста CLI соблюдай: у каждой задачи указан способ проверки, последняя группа задач — make lint и make test.
5. tasks.md: 3–6 групп задач, у каждой группы свой шаг проверки; задачи атомарные.
6. Без git commit/push. Если что-то невозможно — скажи прямо в ответе, обходных выдумок не делай.`;

const APPLY_SYSTEM = `Ты — реализатор изменений OpenSpec в проекте skvoznoi_project (веха M1 «Каркас»).
1. Прочитай файл .zcode/skills/openspec-apply-change/SKILL.md и следуй ему.
2. Работай в корне проекта; используй .venv/bin/python и .venv/bin/pytest; для make и pytest задавай Bash timeout 300000+ мс.
3. Задачи tasks.md выполняй строго по группам в порядке следования; проверку группы прогоняй сам; помечай задачи [x] сразу после её выполнения и проверки.
4. Комментарии и docstring — на русском, идентификаторы — на английском; изменения минимальные и только в рамках change'а.
5. Без git commit/push; не трогай файлы вне каталога проекта.
6. Если задача невозможна или инструкции противоречат — скажи прямо в ответе, не имитируй успех.`;

const VERIFY_SYSTEM = `Ты — независимый верификатор изменений OpenSpec в проекте skvoznoi_project (веха M1 «Каркас»).
1. Прочитай файл .zcode/skills/openspec-verify-change/SKILL.md и следуй ему.
2. Код не исправляй, артефакты не переписывай — только диагностика.
3. Свежие прогоны: .venv/bin/python -m pytest по тест-файлам change'а (Bash timeout 300000+ мс), make lint, openspec validate <имя> --strict, плюс приёмочные проверки из спеки и tasks (import, демо-скрипты и т.п.).
4. Отчёт: .opencode/queue/verify/<имя>.md — первая строка «ВЕРДИКТ: PASS» или «ВЕРДИКТ: FAIL», далее критичные (блокируют архивацию) и некритичные замечания со ссылками файл:строка.
5. Вердикт обосновывай результатами прогонов, не мнением. Без git commit/push.`;

const outcomes: ChangeOutcome[] = [];
const verified: string[] = [];
const notCovered: string[] = [];
const findings: Finding[] = [];
let failed: { change: string; stage: string; notes: string } | null = null;

for (const c of CHANGES) {
  if (failed !== null) break;

  // ---------- 1. Proposal ----------
  phase("Создать proposal change'а");
  log(`Изменение ${c.name}: создаю planning-артефакты по плану M1.`);
  const proposer = agent(`план-${c.name}`, { model: FLASH, system: PROPOSE_SYSTEM });
  let proposal: ProposalResult | null = null;
  let proposalErr = "change ещё не создавался";
  for (let round = 1; round <= 3; round += 1) {
    proposal = await proposer.ask<ProposalResult>(
      `Создай change «${c.name}» вехи M1 по docs/plan-m1-foundation.md. Твоя строка — №${c.row} таблицы §4. Содержание: ${c.scope}. Способ проверки: ${c.check}. Дополнительно учти: скелет config.yaml из §5 (для m1-config), схему SQLite из §6 (для m1-storage), решения §7, DoD §8. Порядок: openspec new change ${c.name} → status --json → артефакты по зависимостям → validate. Текущее состояние: ${proposalErr}. Верни change, artifacts, summary.`,
    );
    const pv = await world.run("openspec", ["validate", c.name, "--strict"]);
    if (pv.exitCode === 0) {
      verified.push(`openspec validate ${c.name} --strict — PASS`);
      break;
    }
    proposalErr = `валидация упала: ${(pv.stderr.trim() || pv.stdout.trim()).slice(0, 1500)}`;
    proposal = null;
  }
  if (proposal === null) {
    failed = { change: c.name, stage: "proposal", notes: proposalErr };
    break;
  }
  report({ change: c.name, status: "proposal готов", notes: proposal.summary }, "conveyor");

  // ---------- 2. Apply ----------
  phase("Реализовать задачи change'а");
  const implementer = agent(`реализатор-${c.name}`, { model: FLASH, system: APPLY_SYSTEM });
  let groupsDone = 0;
  let applyDeviations = "";
  let runaway = true;
  for (let g = 1; g <= 12; g += 1) {
    const step = await implementer.ask<ApplyStep>(
      `Выполни СЛЕДУЮЩУЮ группу задач из tasks.md change'а «${c.name}» (группа №${g}).${g === 1 ? " Начни с openspec instructions apply --change " + c.name + " --json и чтения контекстных файлов (proposal, specs, design, tasks)." : ""} После группы прогони её проверку сам, убедись, что она зелёная, и пометь выполненные задачи [x]. Верни summary, testFiles (тест-файлы группы, иначе []), moreGroups, deviations.`,
    );
    if (step.testFiles.length > 0) {
      let pytestRun = await world.run(".venv/bin/python", ["-m", "pytest", ...step.testFiles], { timeoutMs: 300000 });
      let round = 1;
      while (pytestRun.exitCode !== 0 && round < 3) {
        await implementer.ask(
          `Scoped-проверка группы №${g} change'а «${c.name}» упала:\n\n${(pytestRun.stderr.trim() || pytestRun.stdout.trim()).slice(0, 3000)}\n\nИсправь код или тесты этой группы, добейся зелёного прогона сам и верни одну строку — что исправлено.`,
        );
        pytestRun = await world.run(".venv/bin/python", ["-m", "pytest", ...step.testFiles], { timeoutMs: 300000 });
        round += 1;
      }
      if (pytestRun.exitCode !== 0) {
        failed = {
          change: c.name,
          stage: `apply, группа №${g}`,
          notes: `scoped-pytest не зелёный после 3 попыток: ${(pytestRun.stderr.trim() || pytestRun.stdout.trim()).slice(0, 800)}`,
        };
        break;
      }
      verified.push(`scoped pytest (${step.testFiles.join(", ")}) — PASS`);
    }
    groupsDone += 1;
    if (step.deviations.length > 0) applyDeviations += `${step.deviations} `;
    if (!step.moreGroups) {
      runaway = false;
      break;
    }
  }
  if (failed !== null) break;
  if (runaway) {
    failed = { change: c.name, stage: "apply", notes: "более 12 групп задач в tasks.md — цикл остановлен" };
    break;
  }
  if (applyDeviations.trim().length > 0) {
    findings.push({
      where: c.name,
      what: `Отклонения от артефактов при реализации: ${applyDeviations.trim()}`,
      evidence: "отчёты реализатора по группам задач",
      status: "unconfirmed",
      severity: "low",
    });
  }
  report({ change: c.name, status: "apply готов", notes: `${groupsDone} групп задач выполнено` }, "conveyor");

  // ---------- 3. Make-ворота ----------
  phase("Прогнать ворота make install, lint, test");
  const makeGates: Array<{ target: string; timeoutMs: number }> = [
    { target: "install", timeoutMs: 300000 },
    { target: "lint", timeoutMs: 300000 },
    { target: "test", timeoutMs: 600000 },
  ];
  for (const mg of makeGates) {
    let makeRun = await world.run("make", [mg.target], { timeoutMs: mg.timeoutMs });
    let round = 1;
    while (makeRun.exitCode !== 0 && round < 3) {
      await implementer.ask(
        `make ${mg.target} упал в change'е «${c.name}»:\n\n${(makeRun.stderr.trim() || makeRun.stdout.trim()).slice(0, 3000)}\n\nИсправь, добейся зелёного прогона и верни одну строку — что исправлено.`,
      );
      makeRun = await world.run("make", [mg.target], { timeoutMs: mg.timeoutMs });
      round += 1;
    }
    if (makeRun.exitCode !== 0) {
      failed = {
        change: c.name,
        stage: `make ${mg.target}`,
        notes: (makeRun.stderr.trim() || makeRun.stdout.trim()).slice(0, 800),
      };
      break;
    }
    verified.push(`make ${mg.target} (${c.name}) — PASS`);
  }
  if (failed !== null) break;

  // ---------- 4. Verify ----------
  phase("Независимая проверка change'а");
  const verifier = agent(`верификатор-${c.name}`, { model: MAX, system: VERIFY_SYSTEM });
  let vr = await verifier.ask<VerifyResult>(
    `Проверь реализацию change'а «${c.name}» (артефакты в openspec/changes/${c.name}/). Ворота make install/lint/test уже пройдены скриптом — они зелёные, дублировать полный прогон не нужно. Сделай свои свежие прогоны scoped: pytest по тест-файлам change'а, openspec validate ${c.name} --strict, приёмочные проверки из спеки и tasks (import, демо-скрипты и т.п.). Отчёт запиши в .opencode/queue/verify/${c.name}.md (первая строка «ВЕРДИКТ: …»). Верни verdict, critical, reportPath.`,
  );
  let rework = 0;
  while (vr.verdict === "FAIL" && rework < 3) {
    log(`Изменение ${c.name}: верификатор отклонил, доработка ${rework + 1} из 3.`);
    const crit = vr.critical.length > 0 ? vr.critical.join("\n- ") : `см. отчёт ${vr.reportPath}`;
    await implementer.ask(
      `Верификатор отклонил change «${c.name}». Критичные замечания:\n- ${crit}\n\nИсправь их в коде, прогони затронутые тесты сам и верни одну строку — что исправлено.`,
    );
    const reRun = await world.run("make", ["test"], { timeoutMs: 600000 });
    if (reRun.exitCode !== 0) {
      failed = {
        change: c.name,
        stage: "make test (после доработки по верификатору)",
        notes: (reRun.stderr.trim() || reRun.stdout.trim()).slice(0, 800),
      };
      break;
    }
    vr = await verifier.ask<VerifyResult>(
      `Замечания исправлены, make test зелёный. Перепроверь change «${c.name}» по прежним правилам, обнови отчёт .opencode/queue/verify/${c.name}.md и верни обновлённые verdict, critical, reportPath.`,
    );
    rework += 1;
  }
  if (failed !== null) break;
  if (vr.verdict !== "PASS") {
    failed = {
      change: c.name,
      stage: "verify",
      notes: `FAIL после ${rework} доработок; критичное: ${(vr.critical.join("; ") || "см. отчёт " + vr.reportPath).slice(0, 500)}`,
    };
    break;
  }
  verified.push(`независимая проверка ${c.name} (GLM-5.3 max) — PASS, отчёт: ${vr.reportPath}`);
  try {
    await artifact.file("verify-report", vr.reportPath, {
      title: `Отчёт верификации: ${c.name}`,
      description: "Независимая проверка реализации против артефактов change'а.",
    });
  } catch {
    log(`Не удалось опубликовать отчёт верификации ${c.name} (${vr.reportPath}).`);
  }
  report({ change: c.name, status: "verify PASS", notes: `отчёт: ${vr.reportPath}` }, "conveyor");

  // ---------- 5. Archive ----------
  phase("Архивация change'а");
  const arch = await world.run("openspec", ["archive", c.name, "-y"]);
  if (arch.exitCode !== 0) {
    failed = {
      change: c.name,
      stage: "archive",
      notes: (arch.stderr.trim() || arch.stdout.trim()).slice(0, 800),
    };
    break;
  }
  verified.push(`openspec archive ${c.name} -y — выполнен`);
  report({ change: c.name, status: "архивирован", notes: `verify PASS, отчёт: ${vr.reportPath}` }, "conveyor");
  outcomes.push({ change: c.name, status: "архивирован", notes: `${groupsDone} групп задач; ${vr.reportPath}` });
  log(`Изменение ${c.name} завершено: proposal, apply (${groupsDone} групп), verify PASS, заархивировано.`);
}

// Итог конвейера: провал, остановка и неохваченные change'ы.
if (failed !== null) {
  report({ change: failed.change, status: "ПРОВАЛЕН", notes: `${failed.stage}: ${failed.notes}` }, "conveyor");
  findings.push({
    where: failed.change,
    what: `Конвейер остановлен на этапе «${failed.stage}»`,
    evidence: failed.notes,
    status: "verified",
    severity: "high",
  });
  let failIdx = -1;
  for (let i = 0; i < CHANGES.length; i += 1) {
    if (CHANGES[i].name === failed.change) failIdx = i;
  }
  for (let i = failIdx; i < CHANGES.length; i += 1) {
    notCovered.push(i === failIdx ? `${CHANGES[i].name} — не завершён (остановлено на «${failed.stage}»)` : `${CHANGES[i].name} — не начинался`);
  }
  outcomes.push({ change: failed.change, status: "ПРОВАЛЕН", notes: failed.stage });
}

const archivedCount = outcomes.filter((o) => o.status === "архивирован").length;

const lines: string[] = ["# Конвейер вехи M1 — итог", ""];
if (failed === null) {
  lines.push(`Все ${CHANGES.length} change'а проведены через конвейер: proposal → apply → make-ворота → независимая проверка → архивация.`, "");
} else {
  lines.push(`Конвейер остановлен на change'е «${failed.change}», этап «${failed.stage}».`, "", `Причина: ${failed.notes}`, "");
}
lines.push("## Изменения", "");
for (const o of outcomes) {
  lines.push(`- **${o.change}** — ${o.status}. ${o.notes}`);
}
if (findings.length > 0) {
  lines.push("", "## Замечания", "");
  for (const f of findings) lines.push(`- [${f.severity}/${f.status}] ${f.where}: ${f.what}`);
}
if (verified.length > 0) {
  lines.push("", "## Прогоны и проверки", "");
  for (const v of verified) lines.push(`- ${v}`);
}
if (notCovered.length > 0) {
  lines.push("", "## Не покрыто", "");
  for (const n of notCovered) lines.push(`- ${n}`);
}

await artifact.markdown("report", lines.join("\n"), {
  title: "Отчёт конвейера M1",
  description:
    failed === null
      ? "Все change'а вехи M1 реализованы, проверены и заархивированы."
      : `Остановлено на «${failed.change}» (этап «${failed.stage}»); заархивировано ${archivedCount} из ${CHANGES.length}.`,
  primary: true,
});

const conclusion =
  failed === null
    ? `Конвейер plan-m1 завершён полностью: ${archivedCount} change'а (m1-tooling, m1-common, m1-config, m1-storage) прошли proposal, реализацию по группам, make-ворота (install/lint/test), независимую проверку GLM-5.3 max и заархивированы. Критерий вехи «make install && make lint && make test» зелёный.`
    : `Конвейер plan-m1 остановлен на change'е «${failed.change}», этап «${failed.stage}». Заархивировано ${archivedCount} из ${CHANGES.length}. Причина: ${failed.notes}`;

return { conclusion, findings, verified, notCovered };