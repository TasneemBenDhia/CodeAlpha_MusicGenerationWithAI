# AI Music Generation using Recurrent Neural Networks (LSTM)

An end-to-end Deep Learning project designed to process raw musical data (MIDI format) and train a generative Recurrent Neural Network (RNN). Utilizing **Long Short-Term Memory (LSTM)** layers, the model learns the structural grammar, pitch transitions, and harmonic dependencies of input compositions to autonomously synthesize entirely new, cohesive musical arrangements.

---

## 📌 Project Overview & Objectives

In deep learning, music can be treated as a time-series or sequential data stream where a given note heavily relies on the context of the notes that preceded it. This pipeline mimics that human cognitive process through the following steps:

1. **Sequential Data Extraction:** Parse polyphonic and monophonic MIDI compositions to extract individual note pitches, durations, and concurrent structural groupings (Chords).
2. **Feature Engineering:** Translate raw musical characters into standardized integer tokens and scale them into normalized inputs suitable for a neural network.
3. **Architecture Optimization:** Design an LSTM network with dropout regularization capable of storing long-range contextual history without suffering from vanishing gradients.
4. **Predictive Modeling:** Execute multi-epoch training loops to shift a probability distribution (via Softmax) closer to the actual note patterns found in the dataset.
5. **Generative Sampling:** Seed the fully trained network with a random sequence and loop a rolling window prediction mechanism to write a novel `.mid` file from scratch.

---

## 🧠 Core Deep Learning Concepts Explained

### 1. Why use an LSTM Network for Music?
Standard Feedforward Neural Networks treat inputs independently, making them poor choices for music where context is everything. Standard Recurrent Neural Networks (RNNs) can remember previous tokens but suffer from a math flaw called the **Vanishing Gradient Problem**, meaning they forget things that happened more than a few steps back.

**Long Short-Term Memory (LSTM)** architectures solve this by utilizing specific logical structures called "gates" (Input, Forget, and Output Gates). These gates regulate the flow of information, allowing the network to retain a stable memory of musical themes, motifs, and rhythmic structures across long sequences (e.g., maintaining a specific musical key across 50+ time steps).

### 2. Time-Series Windowing
Your model uses a lookback window of **50 time steps**. This means that to predict note number 51, the model reviews the pattern of the previous 50 notes. As it generates new music, it slides this 50-note window forward by one step, feeding its own previous prediction back into the network as the newest context token.

---

## 📁 Repository Directory Structure

```text
music_generation_project/
│
├── midi_songs/             # Target directory for your training dataset
│   ├── song1.mid           # e.g., Chopin Nocturnes, Bach Fugues, etc.
│   ├── song2.mid
│   └── song3.mid
│
├── preprocess.py           # Module: Extracts MIDI data, processes tokens, handles windowing
├── train.py                # Module: Defines Keras Sequential layers and sets callbacks
├── generate.py             # Module: Manages random seeding and rolling prediction arrays
├── run.py                  # Orchestration: Executes Phases 1-4 seamlessly in sequence
└── README.md               # Documentation: Project guide and system details