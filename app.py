import gradio as gr
from src.predict import EmotionPredictor

# Initialize the predictor (it automatically loads models from the outputs/ directory)
predictor = EmotionPredictor(model_dir='outputs')

def analyze_emotion(text):
    if not text.strip():
        return {"NEUTRAL": 1.0}
    
    # Predict the emotion using our local pipeline
    emotion, conf, probs = predictor.predict(text)
    
    # Gradio's Label component expects a dictionary of {class_name: probability}
    # We round the probabilities for a cleaner UI
    return {k: float(v) for k, v in probs.items()}

# Build the Gradio interface using modern Blocks API
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 600px; margin: 0 auto;">
            <h1>🧠 AI Emotion Detection Prototype</h1>
            <p>Type a sentence below and our NLP model will predict the underlying emotional state across six categories.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text", 
                placeholder="e.g., I'm absolutely thrilled about my new job offer!",
                lines=4
            )
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
            
            # Add some examples for users to click on
            gr.Examples(
                examples=[
                    "I'm absolutely thrilled about my new job offer!",
                    "I'm crying because I miss her so much it hurts...",
                    "This incompetent service is making me lose my mind!",
                    "I can't believe they threw me a surprise party!",
                    "Please forward the meeting notes to the team by Friday."
                ],
                inputs=text_input
            )
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=6, label="Emotion Confidence Scores")
            
    # Trigger prediction on button click
    submit_btn.click(fn=analyze_emotion, inputs=text_input, outputs=output_label)

if __name__ == "__main__":
    demo.launch(share=False)
