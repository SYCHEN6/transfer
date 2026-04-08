---
name: gitmoji_commit
description: Analyzes staged code changes and conversation context to generate a Gitmoji-compliant commit message in English, then executes the commit locally (without pushing).
---
# Gitmoji Commit Skill

This Skill automates the Git commit process. You will analyze code changes and current conversation context (user instructions and your modifications) to generate a commit message following the [Gitmoji](https://gitmoji.dev/) specification.

## Core Principles

1. **Always use ENGLISH** for the commit Subject and Body.
2. **STRICTLY NO PUSHING (`git push`)**, only execute local commits (`git commit`).
3. **Formatting Standard**: `<emoji> <type>(<scope>): <subject>`
   * **Emoji**: Use the Gitmoji code (e.g., `:sparkles:`) for better terminal compatibility, rather than the raw Unicode character.
   * **Subject**: A concise English description, starting with an imperative verb (e.g., Add, Fix, Update), limited to 50 characters. This acts as the first `-m` title.
   * **Body (Optional)**: **Strongly recommend using a single `-m` parameter** unless the commit involves massive, multi-layered changes where the title alone cannot convey the core intent. If the `Subject` sufficiently summarizes the intent, keep it simple.
   * *Example (Default/Concise)*: `git commit -m ":sparkles: feat(auth): add user login functionality"`
   * *Example (Complex/Split)*: `git commit -m ":bug: fix(nav): resolve navbar alignment issue" -m "1. Fix margin calculation bug on mobile\n2. Standardize z-index for glassmorphism components"`

## Execution Steps

### 1. Check Staging Area

First, verify if there are any staged files:

```powershell
git diff --cached --name-only
```

* **If empty**: Prompt the user: "The staging area is empty. Please stage your files using `git add` first," and halt execution.
* **If staged files exist**: Proceed to the next step.

### 2. Analyze Changes & Context

Run `git diff --cached` to get the detailed diff. You **MUST** review the current conversation history:

- What did the user just request?
- What are the specific code changes?
- If there are multiple changes, what is the **primary intent**? (e.g., If a script was built and a minor typo was fixed, prioritize the script build. By default, generate a single commit capturing the primary intent).

### 3. Generate Commit Message

Select the **most accurate** Emoji and Type from the reference table based on your analysis.

#### Gitmoji Reference

| Emoji  | Code                            | Description                                                   |
| :----- | :------------------------------ | :------------------------------------------------------------ |
| 🎨     | `:art:`                       | Improve structure / format of the code.                       |
| ⚡️   | `:zap:`                       | Improve performance.                                          |
| 🔥     | `:fire:`                      | Remove code or files.                                         |
| 🐛     | `:bug:`                       | Fix a bug.                                                    |
| 🚑️   | `:ambulance:`                 | Critical hotfix.                                              |
| ✨     | `:sparkles:`                  | Introduce new features.                                       |
| 📝     | `:memo:`                      | Add or update documentation.                                  |
| 🚀     | `:rocket:`                    | Deploy stuff.                                                 |
| 💄     | `:lipstick:`                  | Add or update the UI and style files.                         |
| 🎉     | `:tada:`                      | Begin a project.                                              |
| ✅     | `:white_check_mark:`          | Add, update, or pass tests.                                   |
| 🔒     | `:lock:`                      | Fix security or privacy issues.                               |
| 🔐     | `:closed_lock_with_key:`      | Add or update secrets.                                        |
| 🔖     | `:bookmark:`                  | Release / Version tags.                                       |
| 🚨     | `:rotating_light:`            | Fix compiler / linter warnings.                               |
| 🚧     | `:construction:`              | Work in progress.                                             |
| 💚     | `:green_heart:`               | Fix CI Build.                                                 |
| ⬇️   | `:arrow_down:`                | Downgrade dependencies.                                       |
| ⬆️   | `:arrow_up:`                  | Upgrade dependencies.                                         |
| 📌     | `:pushpin:`                   | Pin dependencies to specific versions.                        |
| 👷     | `:construction_worker:`       | Add or update CI build system.                                |
| 📈     | `:chart_with_upwards_trend:`  | Add or update analytics or track code.                        |
| ♻️   | `:recycle:`                   | Refactor code.                                                |
| ➕     | `:heavy_plus_sign:`           | Add a dependency.                                             |
| ➖     | `:heavy_minus_sign:`          | Remove a dependency.                                          |
| 🔧     | `:wrench:`                    | Add or update configuration files.                            |
| 🔨     | `:hammer:`                    | Add or update development scripts.                            |
| 🌐     | `:globe_with_meridians:`      | Internationalization and localization.                        |
| ✏️   | `:pencil2:`                   | Fix typos.                                                    |
| 💩     | `:poop:`                      | Write bad code that needs to be improved.                     |
| ⏪     | `:rewind:`                    | Revert changes.                                               |
| 🔀     | `:twisted_rightwards_arrows:` | Merge branches.                                               |
| 📦️   | `:package:`                   | Add or update compiled files or packages.                     |
| 👽️   | `:alien:`                     | Update code due to external API changes.                      |
| 🚚     | `:truck:`                     | Move or rename resources (e.g.: files, paths, routes).        |
| 📄     | `:page_facing_up:`            | Add or update license.                                        |
| 💥     | `:boom:`                      | Introduce breaking changes.                                   |
| 🍱     | `:bento:`                     | Add or update assets.                                         |
| ♿️   | `:wheelchair:`                | Improve accessibility.                                        |
| 💡     | `:bulb:`                      | Add or update comments in source code.                        |
| 🍻     | `:beers:`                     | Write code drunkenly.                                         |
| 💬     | `:speech_balloon:`            | Add or update text and literals.                              |
| 🗃️   | `:card_file_box:`             | Perform database related changes.                             |
| 🔊     | `:loud_sound:`                | Add or update logs.                                           |
| 🔇     | `:mute:`                      | Remove logs.                                                  |
| 👥     | `:busts_in_silhouette:`       | Add or update contributor(s).                                 |
| 🚸     | `:children_crossing:`         | Improve user experience / usability.                          |
| 🏗️   | `:building_construction:`     | Make architectural changes.                                   |
| 📱     | `:iphone:`                    | Work on responsive design.                                    |
| 🤡     | `:clown_face:`                | Mock things.                                                  |
| 🥚     | `:egg:`                       | Add or update an easter egg.                                  |
| 🙈     | `:see_no_evil:`               | Add or update a .gitignore file.                              |
| 📸     | `:camera_flash:`              | Add or update snapshots.                                      |
| ⚗️   | `:alembic:`                   | Perform experiments.                                          |
| 🔍     | `:mag:`                       | Improve SEO.                                                  |
| 🏷️   | `:label:`                     | Add or update types.                                          |
| 🌱     | `:seedling:`                  | Add or update seed files.                                     |
| 🚩     | `:triangular_flag_on_post:`   | Add, update, or remove feature flags.                         |
| 🥅     | `:goal_net:`                  | Catch errors.                                                 |
| 💫     | `:dizzy:`                     | Add or update animations and transitions.                     |
| 🗑️   | `:wastebasket:`               | Deprecate code that needs to be cleaned up.                   |
| 🛂     | `:passport_control:`          | Work on code related to authorization, roles and permissions. |
| 🩹     | `:adhesive_bandage:`          | Simple fix for a non-critical issue.                          |
| 🧐     | `:monocle_face:`              | Data exploration/inspection.                                  |
| ⚰️   | `:coffin:`                    | Remove dead code.                                             |
| 🧪     | `:test_tube:`                 | Add a failing test.                                           |
| 👔     | `:necktie:`                   | Add or update business logic.                                 |
| 🩺     | `:stethoscope:`               | Add or update healthcheck.                                    |
| 🧱     | `:bricks:`                    | Infrastructure related changes.                               |
| 🧑‍💻 | `:technologist:`              | Improve developer experience.                                 |
| 💸     | `:money_with_wings:`          | Add sponsorships or money related infrastructure.             |
| 🧵     | `:thread:`                    | Add or update code related to multithreading or concurrency.  |
| 🦺     | `:safety_vest:`               | Add or update code related to validation.                     |
| ✈️   | `:airplane:`                  | Improve offline support.                                      |
| 🦖     | `:t-rex:`                     | Code that adds backwards compatibility.                       |

*(Agent Note: Prioritize the most specific context when choosing. E.g., if updating version in `package.json`, `:arrow_up:` or `:heavy_plus_sign:` is better than the generic `:package:` or `:wrench:`. If purely tweaking CSS, you must use `:lipstick:`)*

### 4. User Interaction & Execution

You **MUST** display the proposed commit command to the user and request confirmation first.

**Example Conversation**:

> **Agent**: The staging area contains dependency updates in `package.json`.
> Proposed commit message:
> `git commit -m ":arrow_up: chore(deps): upgrade vue version to 3.4"`
>
> Shall I execute this?

**ONLY after the user explicitly replies with "Yes", "Confirm", "OK", etc.,** proceed to run the command:

```powershell
# Simple Commit
git commit -m ":your-emoji: type(scope): subject"

# Complex Commit
git commit -m ":your-emoji: type(scope): subject" -m "1. Detail one\n2. Detail two"
```

## Important Notes

- If the user is unsatisfied with the proposed message, adjust it based on their feedback and ask for confirmation again.
- Once successfully committed, inform the user and remind them to manually push the changes.
