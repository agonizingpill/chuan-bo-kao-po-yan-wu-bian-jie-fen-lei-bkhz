# 初始 schema

`src.models.Base.metadata.create_all()` 按外键顺序建表，由 `src.main` 在 API 首次启动时执行。已有开发卷不执行自动字段迁移。项目 README 记录了模型与首次建库入口。

## 模型版本治理（model_versions 与案件版本绑定）

新建库由 `create_all()` 直接包含 `model_versions` 表及 `delay_incidents.model_version`、`recompute`、`recompute_error`、`recompute_at` 列。已存在的开发卷需要先备份卷，再执行以下 DDL（MySQL 8）：

```sql
CREATE TABLE IF NOT EXISTS model_versions (
  code VARCHAR(32) NOT NULL PRIMARY KEY,
  state ENUM('staged','active','retired','damaged') NOT NULL,
  seed INTEGER NOT NULL,
  parameters JSON NOT NULL,
  metrics JSON NOT NULL,
  sklearn_version VARCHAR(32) NOT NULL,
  artifact_path VARCHAR(200) NOT NULL,
  artifact_digest VARCHAR(64) NOT NULL,
  note VARCHAR(300) NOT NULL DEFAULT '',
  created_by VARCHAR(32) NOT NULL,
  created_at DATETIME NOT NULL,
  activated_at DATETIME NULL,
  CONSTRAINT fk_model_versions_user FOREIGN KEY (created_by) REFERENCES users (username)
);
CREATE INDEX ix_model_versions_state ON model_versions (state);

ALTER TABLE delay_incidents
  ADD COLUMN model_version VARCHAR(32) NOT NULL DEFAULT 'v1',
  ADD COLUMN recompute JSON NULL,
  ADD COLUMN recompute_error VARCHAR(180) NULL,
  ADD COLUMN recompute_at DATETIME NULL;
```

迁移后首次启动会由 `ensure_baseline()` 在产物目录留存并激活基线模型 `v1`；历史案件的 `model_version` 默认即为 `v1`，与原固定种子模型口径一致。

