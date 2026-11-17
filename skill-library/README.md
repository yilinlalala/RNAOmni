# skill-library

过夜技能注册与规程。阶段状态在 [`../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md`](../devlog/20260719_000000_feature_skill_PLAN_PARTIAL.md)。

| 文档 | 说明 |
|------|------|
| [registry.yaml](registry.yaml) | 技能注册（launcher 只读这里） |
| [feature/SKILL.md](feature/SKILL.md) | 竖切入口：人控特征；agent 只改 recipe 白名单 + `train.py` |
| [../tools/resolve_skills.py](../tools/resolve_skills.py) | 解析注册表并生成 prompt 块 |

```bash
python tools/resolve_skills.py --overnight --prompt
```
