import streamlit as st
import time
from tokenizer import BytePairEncoder
from pathlib import Path
import json 

st.set_page_config(page_title="BPE Visualizer", layout="wide")
st.title("BPE Tokenizer Visualizer")

if "bpe" not in st.session_state:
    st.session_state.bpe = None
if "encoded_tokens" not in st.session_state:
    st.session_state.encoded_tokens = None
if "original_text" not in st.session_state:
    st.session_state.original_text = ""
if "training_option" not in st.session_state:
    st.session_state.training_option = ""
    
text_area_content = {
    "wikipedia-en-sample.txt": "The mistralet or magistralou is a moderate, pleasant, and favorable mistral wind.",
    "wikipedia-fr-sample.txt": "Le mistralet ou magistralou, est un mistral modéré, agréable et favorable.",
    "python-code-sample.txt": """def ask_input(message: str) -> str:
    _check_no_input(message)
    return input(message)"""
}

st.header("Configuration")
st.markdown("Tout d'abord, vous devez **configurer le BPE** et choisir sur **quelles données vous voulez entrainer le model** et la **taille du vocabulaire** attendu.")
    
options = {f.name:f for f in(Path(__file__).resolve().parents[1] / "data").iterdir() if f.is_file()}
training_option = st.pills("Données d'entraînement: *", options.keys())
if training_option == None:
    st.session_state.training_option = training_option

if st.session_state.training_option != training_option:
    st.session_state.training_option = training_option
    st.session_state.bpe = None
    
vocab_size = st.slider(
    "Taille du vocabulaire: *",
    min_value=257,
    max_value=2000,
    value=1000,
    step=50
)
    
# Bouton train
if st.button("Entraîner le BPE", use_container_width=True):
    if training_option != None:
        with open(options[training_option], "r") as file:
            training_text = file.read()
               
        with st.spinner("Entraînement en cours..."):
            st.session_state.bpe = BytePairEncoder()
            st.session_state.bpe.train(training_text, vocab_size)
        st.success("BPE entraîné avec succès!") 
    else:
        st.error("Veuillez séléctionner les données d'entraînement")
    
if st.session_state.bpe is not None:
    st.divider()
    col1, col2 = st.columns([1, 1], gap="large")
    
    if training_option in text_area_content:
        content = text_area_content[training_option]
    else:
        content = "Enter the text you want to encode."
    
    with col1:
        col_name, col_select = st.columns([3, 2], gap="large")
        with col_name:
            st.markdown("<h3 style='padding:0'>Texte à encoder</h3>", unsafe_allow_html=True)
        with col_select:
            on = st.toggle("Conversation mode")
        
        if 'lines' not in st.session_state:
            st.session_state.lines = [
                {"role": "System", "message": "Tu es un assistant"}, 
                {"role": "User", "message": "Qu'est ce que c'est un mistral ?"},
                {"role": "Assistant", "message": "Le mistral est un vent du nord catabatique"}
            ]

        def delete_line(index):
            st.session_state.lines.pop(index)

        if on:
            for i in range(len(st.session_state.lines)):
                col_role, col_message = st.columns([1, 2])
                
                with col_role:
                    st.session_state.lines[i]["role"] = st.selectbox(
                        "Role :", 
                        ("System", "User", "Assistant"), 
                        index=["System", "User", "Assistant"].index(st.session_state.lines[i]["role"]),
                        key=f"role_{i}",
                        label_visibility="collapsed"
                    )
                
                with col_message:
                    st.session_state.lines[i]["message"] = st.text_input(
                        "Content", 
                        value=st.session_state.lines[i]["message"],
                        key=f"text_{i}",
                        label_visibility="collapsed"
                    )

            encoded_content = st.session_state.bpe.encode(st.session_state.lines, "conversation")
            content = st.session_state.bpe.decode(encoded_content)
            
            user_text = st.text_area(
                "Entre ton texte:",
                value=content,
                height="stretch",
                label_visibility="collapsed",
                disabled=True
            )
        else:
            user_text = st.text_area(
                "Entre ton texte:",
                value=content,
                height="stretch",
                label_visibility="collapsed"
            )
                
        if len(user_text) < 1:
                user_text["value"] = content
        
    
    with col2:
        st.subheader("Tokens visualisés")
        if on:
            mode = "conversation"
            
            try:
                data_to_encode = json.loads(user_text)
            except:
                data_to_encode = st.session_state.lines
        else:
            mode = "document"
            data_to_encode = user_text
            
        if user_text:
            encoded = st.session_state.bpe.encode(data_to_encode, mode)
            
            html_tokens = []
            for token_id in encoded:
                hue = (token_id * 137.5) % 360
                saturation = 70
                lightness = 50
                color = f"hsl({hue}, {saturation}%, {lightness}%)"
                
                try:
                    token_text = st.session_state.bpe.vocab[token_id].decode('utf-8', errors='strict')
                    token_text = token_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    token_text = (token_text
                                .replace(" ", "␣")
                                .replace("\n", "↵")
                                .replace("\t", "→")
                                .replace("_", "＿"))
                except:
                    token_text = f"[{token_id}]"
                
                html_tokens.append(
                    f'<span style="background-color: {color}; padding: 6px 10px; '
                    f'border-radius: 4px; margin: 3px; display: inline-block; '
                    f'font-weight: 500; font-size: 14px; border: 1px solid rgba(0,0,0,0.1); '
                    f'font-family: monospace;">'  # Ajout de font-family monospace
                    f'{token_text}</span>'
                )
            
            st.markdown("".join(html_tokens), unsafe_allow_html=True)
            st.caption("Légende: ␣ = espace | ↵ = retour ligne | → = tabulation")
            st.caption(f"Token IDs: {encoded}")
            
    ## Stats Section
    st.subheader("Infos rapides")
    if user_text:
        start_time = time.time()
        encoded = st.session_state.bpe.encode(data_to_encode, mode)
        encoding_time = (time.time() - start_time) * 1000
        
        original_bytes = len(user_text.encode('utf-8'))
        num_tokens = len(encoded)
        compression_ratio = original_bytes / num_tokens if num_tokens > 0 else 0
        
        col1, col2, col3, col4 = st.columns([1,1,1,1])
        with col1:
            st.metric("Bytes originaux", original_bytes)
        with col2:
            st.metric("Nombre de tokens", num_tokens)
        with col3:
            st.metric("Ratio compression", f"{compression_ratio:.2f}x")
        with col4:
            st.metric("Temps encoding", f"{encoding_time:.2f}ms")
            
    ## Pair Visualization Section        
    st.divider()
    st.subheader("Historique des merges")
    
    with st.expander("Voir tout les merges de paires", expanded=False):
        if hasattr(st.session_state.bpe, 'merges') and st.session_state.bpe.merges:
            st.markdown(f"**Total de merges effectuées:** {len(st.session_state.bpe.merges)}")
            
            show_all = st.checkbox("Afficher toutes les merges", value=False)
            max_display = len(st.session_state.bpe.merges) if show_all else min(50, len(st.session_state.bpe.merges))
            
            if not show_all:
                st.info(f"Affichage des {max_display} premières merges. Cochez la case ci-dessus pour voir toutes les merges.")
            
            merge_data = []
            for i, (pair, new_token_id) in enumerate(list(st.session_state.bpe.merges.items())[:max_display]):
                try:
                    token1 = st.session_state.bpe.vocab[pair[0]].decode('utf-8', errors='replace')
                    token2 = st.session_state.bpe.vocab[pair[1]].decode('utf-8', errors='replace')
                    merged = st.session_state.bpe.vocab[new_token_id].decode('utf-8', errors='replace')
                    
                    token1_display = token1.replace(" ", "␣").replace("\n", "↵").replace("\t", "→")
                    token2_display = token2.replace(" ", "␣").replace("\n", "↵").replace("\t", "→")
                    merged_display = merged.replace(" ", "␣").replace("\n", "↵").replace("\t", "→")
                    
                    merge_data.append({
                        "Ordre": i + 1,
                        "Token 1": f"`{token1_display}`",
                        "ID 1": pair[0],
                        "Token 2": f"`{token2_display}`",
                        "ID 2": pair[1],
                        "→ Résultat": f"`{merged_display}`",
                        "Nouveau ID": new_token_id
                    })
                except:
                    merge_data.append({
                        "Ordre": i + 1,
                        "Token 1": f"[{pair[0]}]",
                        "ID 1": pair[0],
                        "Token 2": f"[{pair[1]}]",
                        "ID 2": pair[1],
                        "→ Résultat": f"[{new_token_id}]",
                        "Nouveau ID": new_token_id
                    })
            
            for merge in merge_data:
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.markdown(f"**#{merge['Ordre']}**")
                with col2:
                    st.markdown(f"{merge['Token 1']} `[{merge['ID 1']}]` + {merge['Token 2']} `[{merge['ID 2']}]` → {merge['→ Résultat']}")
                with col3:
                    st.markdown(f"`ID: {merge['Nouveau ID']}`")
                
                if merge['Ordre'] < max_display:
                    st.markdown("---")
        else:
            st.warning("Aucune information de merge disponible. Assurez-vous que votre BPE stocke les merges pendant l'entraînement.")

    # Footer
    st.divider()
    st.caption("BPE Tokenizer Visualizer | Adrien Tranchant")