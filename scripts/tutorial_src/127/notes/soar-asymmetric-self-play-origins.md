---
slug: soar-asymmetric-self-play-origins
title: 非對稱 self-play 的來源
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [self-play, asymmetric-self-play, AlphaZero, Alice-Bob]
arxiv: ["2601.18778", "1712.01815", "1703.05407", "2101.04882"]
sources: ["arXiv:2601.18778 §2, §A.2（本文對各工作的描述）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["rl-mdp-q-learning-basics", "soar-curriculum-and-environment-design"]
---
# 非對稱 self-play 的來源

本文 §2、§A.2 以三項工作作為 self-play 的脈絡：

- **AlphaZero**（Silver et al.，Science 362(6419)，2018；arXiv 預印本 1712.01815，2017，題名 Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm）：本文將其作為不依賴外部資料、以 self-play 訓練的遊戲代理之例。MDP 與價值函數的基本記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。
- **Sukhbaatar et al.**（arXiv 1703.05407，ICLR 2018，Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play）：本文描述為 Alice–Bob 框架，Alice 提出任務、Bob 嘗試解決，產生「剛好夠難」的任務序列，形成自動課程。
- **OpenAI et al.**（arXiv 2101.04882，2021，Asymmetric self-play for automatic goal discovery in robotic manipulation）：本文描述為將上述想法擴展到機器人操作，不需人工指定任務即可自動發現多樣的操作目標。

SOAR 的 teacher–student 設計屬於此一非對稱形式：teacher 產生問題，student 學習。本文 §A.2 指出，把這個想法從機器人與控制移到 LLM 會遇到不同的困難：LLM 面對離散的符號問題空間，沒有環境模擬器可評估中間進展；teacher 必須產生完整的任務；數學題的正確性近乎二元，沒有通往部分解的梯度。

課程與環境設計方面的相關工作見 [課程學習與環境設計](soar-curriculum-and-environment-design.html)。
