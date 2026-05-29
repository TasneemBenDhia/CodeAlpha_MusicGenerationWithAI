import pickle
from preprocess import prepare_sequences
from train import build_lstm_model, train_model
from generate import generate_music_sequence, export_to_midi

def main():
    print("--- Phase 1: Data Preprocessing ---")
    X, y, vocab_size, raw_input = prepare_sequences(sequence_length=50)
    
    print("\n--- Phase 2: Building Model Architecture ---")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), vocab_size=vocab_size)
    model.summary()
    
    print("\n--- Phase 3: Model Training ---")
    trained_model = train_model(model, X, y)
    
    print("\n--- Phase 4: Creative Generation ---")
    # Reload pickled mappings to ensure consistency
    with open('pitch_names.pkl', 'rb') as f:
        pitch_names = pickle.load(f)
        
    generated_notes = generate_music_sequence(trained_model, raw_input, pitch_names, vocab_size)
    export_to_midi(generated_notes)

if __name__ == "__main__":
    main()