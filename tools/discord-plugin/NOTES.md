# Discord plugin 本地修改紀錄

修改對象：Claude Code 官方 Discord channel plugin（`claude-plugins-official/discord`，版本 0.0.6），檔案 `server.ts`。

| 檔案 | 內容 |
| --- | --- |
| `server.ts` | 修改後的完整檔案 |
| `forum.patch` | 相對於上游 0.0.6 原檔的 unified diff |
| `LICENSE` | 上游的 Apache License 2.0，依授權條款隨修改後的檔案一起保存 |

## 動機

「研究手記」伺服器的論文整理在 ai-research-topic **論壇頻道**，每個主題是一篇貼文（Discord 的實作是 thread）。上游 plugin 的工具都經過 `fetchTextChannel()`，只接受 text-based 頻道。論壇頻道本身不是 text-based，導致兩個問題：

1. `create_thread` 對論壇頻道回報 `channel … not found or not text-based`，bot 無法開新貼文。
2. `fetch_messages` 同樣被拒絕，bot 看不到論壇裡有哪些貼文、拿不到貼文 ID。要把論文放進「對應主題的貼文」時只能請使用者貼連結，曾因此誤開一篇重複的貼文。

## 方法

兩處修改都在工具的 handler 開頭先判斷 `channel.type === ChannelType.GuildForum`。是論壇就走新分支，不是就照原本的流程，所以一般頻道與 DM 的行為不變。

**權限**：兩個新分支都檢查論壇頻道 ID 是否在 `access.json` 的 `groups` 裡，與原本 `fetchAllowedChannel()` 對一般頻道的 allowlist 規則相同。未列入的論壇一律拒絕。

### 1. `create_thread`：在論壇開新貼文

- 論壇頻道改用 `forum.threads.create({ name, message: { content } })` 建立貼文，回傳貼文 ID，之後用 `reply` 以這個 ID 發文。
- 工具 schema 新增選填參數 `text`，作為貼文的第一則內容。沒有給時用標題。

### 2. `fetch_messages`：列出論壇的貼文

- 論壇頻道改為列出貼文，不讀訊息。`forum.threads.fetchActive()` 取得進行中的貼文，`forum.threads.fetchArchived({ type: 'public', limit: 100, before })` 以封存時間往前翻頁取得已封存的貼文，最多 10 頁（1,000 篇）。
- 每篇輸出一行：建立時間、標題、post id、active/archived、訊息數，依建立時間排序。標題中的換行會被替換掉，避免偽造出額外的列（沿用上游對訊息內容的處理）。
- 取得 post id 後，對它呼叫 `fetch_messages` 就是一般的 thread，會走原本的流程讀取訊息；thread 的權限沿用父頻道。
- 工具描述加上一句說明，讓模型知道對論壇頻道會得到貼文清單。

### 驗證

- `bun build --no-bundle server.ts` 轉譯通過。
- 以相同的 discord.js 呼叫寫一次性腳本，列出 ai-research-topic 論壇的 43 篇貼文（active 與 archived 都有），與論壇實際內容一致。
- `create_thread` 的論壇分支已實際開過貼文（「JEPA 與表徵學習」）。

## 套用方式

plugin 位於 `~/.claude/plugins/cache/claude-plugins-official/discord/<版本>/`。plugin 更新會覆蓋 `server.ts`，需要重新套用：

```sh
cd ~/.claude/plugins/cache/claude-plugins-official/discord/<版本>
patch -p1 --dry-run < <repo>/tools/discord-plugin/forum.patch && patch -p1 < <repo>/tools/discord-plugin/forum.patch
```

上游版本若已改動相同位置，`--dry-run` 會失敗，這時依上面「方法」一節手動移植。套用後要**完整重啟 Claude Code**；`/reload-plugins` 不會重啟 `--channels` 啟動的頻道 server。

修改日期：2026-11-19。
