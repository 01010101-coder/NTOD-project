Напиши спецификации OpenSpec для change {change} в облегчённом формате. Каталог openspec/changes/{change} уже создан — заполни его артефактами proposal.md, specs/<capability>/spec.md, design.md, tasks.md (порядок и формат — по выводу 'openspec instructions <artifact> --change {change}'). Если часть артефактов уже есть — дополни недостающие, готовое не переписывай.

Всё нужное из плана и контракта интерфейсов — ниже, docs/development-plan.md и docs/pipeline-contract.md целиком НЕ читай. Имена файлов, классов, сигнатуры, секции конфига и Make-цели бери ровно из контракта.

{context}

Дополнительно прочитай только: openspec/config.yaml, спеки в openspec/specs/ и openspec/changes/*/specs/ для тех capability, на которые {change} опирается напрямую, и сигнатуры (не реализацию) нужного существующего кода. Образец формата — один файл: openspec/changes/archive/*-add-demand-generator/specs/*/spec.md.

Облегчённый формат:
- proposal.md — коротко: зачем, что меняется, затронутые capability;
- spec.md — требования с SHALL/MUST, у каждого 1–2 сценария '#### Scenario' с WHEN/THEN и проверяемыми критериями; без граничных случаев «на всякий случай»;
- design.md — 3–6 ключевых решений по одной-две строки, без длинных разборов альтернатив;
- tasks.md — группы '## N. ...' по 2–5 задач, каждая группа реализуема за один заход; тесты — быстрые, на маленьких данных или синтетике; последняя группа — make lint и make test.
Если change расширяет уже заархивированную capability — ADDED Requirements в дельте к ней (без ## Purpose); новая capability — с ## Purpose.

Код НЕ пиши, другие каталоги changes и код проекта не трогай. В конце 'openspec validate {change} --strict' должен проходить. Работай автономно, без вопросов. Не упоминай ИИ-инструменты в файлах. Отчёт в конце — 2–3 строки.
