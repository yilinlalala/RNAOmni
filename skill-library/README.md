# skill-library

Overnight skill registry and operating docs for feature work.

| Doc | Role |
| --- | --- |
| [registry.yaml](registry.yaml) | Skill registry (read by launchers) |
| [feature/SKILL.md](feature/SKILL.md) | Feature skill entry point |
| [../tools/resolve_skills.py](../tools/resolve_skills.py) | Resolve registry into prompt blocks |

```bash
python tools/resolve_skills.py --overnight --prompt
```
