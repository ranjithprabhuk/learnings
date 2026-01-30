# Getting Started Guide
## Your First Day - AI/ML Learning Journey

Welcome! This guide will help you set up everything you need to begin your 18-week journey to becoming an AI/ML engineer.

---

## ✅ Prerequisites Checklist

Before you start, make sure you have:

- [ ] **Computer**: Mac M3 Pro (or any computer with 8GB+ RAM)
- [ ] **Time Commitment**: 15-25 hours/week available
- [ ] **Mindset**: Ready to learn, fail, and iterate
- [ ] **Motivation**: Clear career goals (write them down!)

---

## 🛠️ Development Environment Setup

### 1. Install Core Tools

#### Node.js (Required)
```bash
# Check if you have Node.js
node --version  # Should be v18 or higher

# If not installed, download from:
# https://nodejs.org/ (LTS version)

# Or use nvm (recommended):
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18
```

#### Git (Required)
```bash
# Check if you have Git
git --version

# If not installed:
# Mac: Install Xcode Command Line Tools
xcode-select --install

# Or download from: https://git-scm.com/
```

#### Code Editor (Required)
**Recommended: VS Code**
- Download: https://code.visualstudio.com/
- Extensions to install:
  - ESLint
  - Prettier
  - TypeScript
  - GitLens
  - Error Lens
  - Thunder Client (for API testing)

### 2. Clone/Fork This Repository

```bash
# If you cloned this repo:
cd learnings
git pull

# If starting fresh:
mkdir ai-ml-learning
cd ai-ml-learning
git init
# Copy all files from this learning plan into the directory
```

### 3. Install Dependencies

```bash
# Install global tools
npm install -g typescript ts-node nodemon prettier eslint

# Verify installations
tsc --version
ts-node --version
```

---

## 🔑 Get API Keys (Free Tiers)

You'll need these for LLM projects (Phases 3-5):

### OpenAI API (Option 1)
1. Go to https://platform.openai.com/signup
2. Sign up and verify email
3. Go to API Keys section
4. Create new API key
5. **IMPORTANT**: Copy key immediately (you won't see it again!)
6. Add $5-10 credits to account

**Free Tier**: $5 credit for new accounts

### Anthropic API - Claude (Option 2, Recommended)
1. Go to https://console.anthropic.com/
2. Sign up
3. Go to API Keys
4. Create new key
5. Copy and save securely

**Free Tier**: Better than OpenAI for learning

### Hugging Face (Optional but useful)
1. Go to https://huggingface.co/
2. Sign up
3. Settings → Access Tokens
4. Create token

**Free Tier**: Yes, generous limits

### Store API Keys Securely

```bash
# In your project root, create .env file
cp .env.example .env

# Edit .env and add your keys
# NEVER commit .env to git (it's in .gitignore)
```

Example `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HUGGINGFACE_API_KEY=hf_...
```

---

## 📁 Project Structure Overview

```
learnings/
├── README.md                    # Main overview
├── PROGRESS.md                  # Your progress tracker
├── .env                         # API keys (DO NOT COMMIT!)
├── .gitignore                   # Git ignore file
│
├── phase-1-fundamentals/        # Weeks 1-2
│   ├── project-1-linear-regression/
│   └── project-2-kmeans-clustering/
│
├── phase-2-neural-networks/     # Weeks 3-4
├── phase-3-domain-overview/     # Week 5
├── phase-4-llm-applications/    # Weeks 6-7
├── phase-5-ai-agents/           # Week 8
├── phase-6-computer-vision/     # Week 9 (optional)
├── phase-7-production/          # Week 10
├── phase-advanced-training/     # Weeks 11-14
├── phase-career/                # Weeks 15-18
│
├── resources/                   # Learning resources
└── docs/                        # Documentation & guides
```

---

## 🎯 Day 1: Your Action Plan

### Morning (2-3 hours)

1. **Set up environment** (30 mins)
   - [ ] Install Node.js, Git, VS Code
   - [ ] Verify installations
   - [ ] Install VS Code extensions

2. **Set up this repository** (30 mins)
   - [ ] Clone/set up project structure
   - [ ] Install global npm packages
   - [ ] Create .env file

3. **Get API keys** (30 mins)
   - [ ] Sign up for Anthropic/OpenAI
   - [ ] Get API key
   - [ ] Test API key with simple curl command:
   ```bash
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{
       "model": "claude-3-haiku-20240307",
       "max_tokens": 100,
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

4. **Read Phase 1 overview** (30 mins)
   - [ ] Read [phase-1-fundamentals/README.md](../phase-1-fundamentals/README.md)
   - [ ] Understand learning objectives
   - [ ] Review project requirements

### Afternoon (2-3 hours)

5. **Start learning math** (1-2 hours)
   - [ ] Watch 3Blue1Brown Linear Algebra Episode 1
   - [ ] Take notes on key concepts
   - [ ] Don't worry if you don't understand everything!

6. **Set up Project 1** (1 hour)
   - [ ] Navigate to phase-1-fundamentals/project-1-linear-regression
   - [ ] Run `npm install`
   - [ ] Review starter code
   - [ ] Create a plan for implementation

7. **Create accountability system** (30 mins)
   - [ ] Set up learning journal (docs/learning-journal-template.md)
   - [ ] Write down your "why" (why are you doing this?)
   - [ ] Set a schedule (which days, what hours?)
   - [ ] Tell someone about your commitment

---

## 📅 Weekly Schedule Template

Choose a schedule that works for you:

### Option 1: Weekday Evenings (15 hrs/week)
- **Mon, Wed, Fri**: 2 hours (6 hrs)
- **Tue, Thu**: 1.5 hours (3 hrs)
- **Weekend**: 3 hours each day (6 hrs)
- **Total**: 15 hours

### Option 2: Intensive (25 hrs/week)
- **Weekdays**: 2 hours/day (10 hrs)
- **Saturday**: 8 hours
- **Sunday**: 7 hours
- **Total**: 25 hours

### Customize Your Schedule

My weekly schedule:
```
Monday: _____ hours (_____am/pm - _____am/pm)
Tuesday: _____ hours (_____am/pm - _____am/pm)
Wednesday: _____ hours (_____am/pm - _____am/pm)
Thursday: _____ hours (_____am/pm - _____am/pm)
Friday: _____ hours (_____am/pm - _____am/pm)
Saturday: _____ hours (_____am/pm - _____am/pm)
Sunday: _____ hours (_____am/pm - _____am/pm)

Total: _____ hours/week
```

---

## 🎓 Learning Best Practices

### 1. Active Learning
- ❌ **Don't**: Just watch videos passively
- ✅ **Do**: Take notes, pause to think, try coding immediately

### 2. Build to Learn
- ❌ **Don't**: Read documentation for hours
- ✅ **Do**: Build a small project to understand the concept

### 3. Spaced Repetition
- ❌ **Don't**: Cram 8 hours on Sunday
- ✅ **Do**: Study 2 hours/day, 4 days/week

### 4. Teach to Master
- ❌ **Don't**: Keep knowledge to yourself
- ✅ **Do**: Explain concepts to others (blog, tweets, rubber duck)

### 5. Embrace Difficulty
- ❌ **Don't**: Give up when confused
- ✅ **Do**: Struggle is where learning happens

---

## 🚫 Common Beginner Mistakes

### Mistake #1: Tutorial Hell
**Problem**: Watching endless tutorials, never building
**Solution**: 20% learning, 80% building. Set a rule: after every tutorial, build something.

### Mistake #2: Perfectionism
**Problem**: Trying to understand everything perfectly before moving on
**Solution**: Get 80% understanding, move forward, come back later

### Mistake #3: Not Tracking Progress
**Problem**: Can't see improvement, feels like you're not progressing
**Solution**: Update PROGRESS.md weekly, keep a learning journal

### Mistake #4: Skipping Fundamentals
**Problem**: Jumping straight to LLMs without understanding basics
**Solution**: Trust the process - Phase 1 math pays off later

### Mistake #5: Not Sharing Work
**Problem**: Building in isolation, no feedback or accountability
**Solution**: Share every project on GitHub, LinkedIn, Twitter

---

## 📝 Create Your Learning Journal

Create a file: `docs/my-learning-journal.md`

```markdown
# My AI/ML Learning Journey

## Why I'm Doing This

[Write 2-3 paragraphs about your motivation]

## Goals

**Short-term (18 weeks)**:
-
-
-

**Long-term (1-2 years)**:
-
-
-

## Week 1 - [Date]

### What I Learned
-
-

### Challenges
-
-

### Wins
-
-

### Tomorrow
-
-
```

---

## 🤝 Join Communities (Do This Week 1!)

### Discord Servers
1. **LangChain** - https://discord.gg/langchain
2. **OpenAI Developers** - https://discord.gg/openai
3. **Hugging Face** - https://discord.gg/huggingface

### Reddit
1. r/learnmachinelearning
2. r/MachineLearning
3. r/artificial

### Twitter
Follow these accounts:
- @karpathy
- @AndrewYNg
- @anthropicAI
- @OpenAI
- @LangChainAI

---

## 🎯 Success Metrics

Track these throughout your journey:

### Week 1 Success
- [ ] Environment set up
- [ ] API keys working
- [ ] Watched first 3Blue1Brown video
- [ ] Started Project 1
- [ ] Created learning journal
- [ ] Joined 2+ communities

### Month 1 Success
- [ ] Completed Phases 1-2 (6 projects)
- [ ] All projects deployed
- [ ] 1+ blog post written
- [ ] Active in 1+ community
- [ ] Updated LinkedIn profile

### Month 4 Success (End of Journey)
- [ ] 10+ projects completed
- [ ] Production SaaS app live
- [ ] 5+ blog posts
- [ ] Resume ready
- [ ] Applying to jobs

---

## 🔧 Troubleshooting

### "npm install" fails
```bash
# Clear cache
npm cache clean --force

# Delete node_modules and try again
rm -rf node_modules
npm install
```

### TypeScript errors
```bash
# Install TypeScript globally
npm install -g typescript

# Verify
tsc --version
```

### API key not working
```bash
# Check .env file exists
ls -la .env

# Check format (no quotes, no spaces)
cat .env

# Restart terminal to load env vars
```

### Can't find a file
```bash
# Check you're in the right directory
pwd

# List files
ls -la
```

---

## 📞 Need Help?

1. **Check FAQs**: [docs/faq.md](./faq.md)
2. **Troubleshooting Guide**: [resources/troubleshooting.md](../resources/troubleshooting.md)
3. **Ask in communities**: Discord, Reddit
4. **Search GitHub Issues**: For specific libraries

---

## ✅ Final Checklist

Before starting Phase 1:

- [ ] Node.js installed and verified
- [ ] Git installed
- [ ] VS Code set up with extensions
- [ ] This repository cloned/set up
- [ ] .env file created with API keys
- [ ] Tested API key with curl command
- [ ] Read Phase 1 README
- [ ] Watched first Linear Algebra video
- [ ] Created learning journal
- [ ] Set weekly schedule
- [ ] Joined 2+ communities
- [ ] Told someone about your goal (accountability!)
- [ ] Excited and ready to learn!

---

## 🚀 Next Steps

**You're all set!** Now:

1. Read [Phase 1: AI Fundamentals](../phase-1-fundamentals/README.md) in detail
2. Start watching 3Blue1Brown videos
3. Begin implementing Project 1: Linear Regression
4. Update [PROGRESS.md](../PROGRESS.md) weekly

**Remember**: This is a marathon, not a sprint. Consistency beats intensity. Show up every day, even if just for 30 minutes.

**You've got this!** 🎉

---

**Questions?** Review the [FAQ](./faq.md) or ask in community Discord servers.

[← Back to Main README](../README.md) | [→ Start Phase 1](../phase-1-fundamentals/)
