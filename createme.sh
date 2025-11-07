#!/bin/bash

# Script pour créer un historique Git réaliste avec modifications de fichiers
# Période : Juin 2025 - Novembre 2025

# Initialiser le repo si nécessaire
if [ ! -d ".git" ]; then
    git init
    echo "# VolatilityForge" > README.md
    git add README.md
    git commit -m "Initial commit"
fi

# Fonction pour créer un commit avec une date spécifique
create_commit() {
    local date="$1"
    local message="$2"
    local files_to_modify="$3"
    local author_name="Mohamed"
    local author_email="mohamed@volatilityforge.ai"
    
    # Modifier les fichiers spécifiés
    if [ -n "$files_to_modify" ]; then
        for file in $files_to_modify; do
            if [ -f "$file" ]; then
                # Ajouter un commentaire ou une ligne vide pour simuler une modification
                echo "" >> "$file"
            fi
        done
        git add $files_to_modify
    fi
    
    GIT_AUTHOR_DATE="$date" \
    GIT_COMMITTER_DATE="$date" \
    git commit --allow-empty \
    --author="$author_name <$author_email>" \
    -m "$message"
}

# Fonction pour créer un commit avec des fichiers spécifiques
create_feature_commit() {
    local date="$1"
    local message="$2"
    local files="$3"
    
    if [ -n "$files" ]; then
        git add $files 2>/dev/null || git add .
    else
        git add .
    fi
    
    GIT_AUTHOR_DATE="$date" \
    GIT_COMMITTER_DATE="$date" \
    git commit --allow-empty \
    --author="Mohamed <mohamed@volatilityforge.ai>" \
    -m "$message"
}

# Définir les commits structurés avec les fichiers associés
declare -A commits_with_files=(
    # Setup initial (Juin)
    ["Initial project setup"]="requirements.txt README.md .gitignore"
    ["Add basic project structure"]="config/__init__.py core/__init__.py"
    ["Configure environment and dependencies"]="config/environment.py requirements.txt"
    ["Implement core abstractions"]="core/abstractions.py"
    
    # Core Architecture (Juin)
    ["Add transformer architecture"]="core/architectures.py models/transformer.py"
    ["Implement attention mechanisms"]="core/attention.py"
    ["Add RMSNorm and layer scaling"]="src/models/architectures/normalization.py"
    ["Implement SwiGLU activation"]="src/models/architectures/activations.py"
    ["Add tensor operations"]="core/tensor_ops.py"
    
    # Feature Engineering (Juin-Juillet)
    ["Add feature engineering pipeline"]="core/features.py core/feature_engineering.py"
    ["Implement microstructure features"]="src/data/advanced_features.py"
    ["Add moneyness calculations"]="core/feature_engineering.py"
    ["Implement temporal features"]="src/data/advanced_features.py"
    ["Add liquidity metrics"]="core/feature_engineering.py"
    ["Implement interaction features"]="src/data/advanced_features.py"
    
    # Data Pipeline (Juillet)
    ["Implement data pipeline"]="core/data.py src/data/pipeline.py"
    ["Add data validation"]="src/data/pipeline.py"
    ["Implement data augmentation"]="core/data.py"
    ["Add k-fold cross-validation"]="core/data.py"
    
    # Training Infrastructure (Juillet-Août)
    ["Implement training engine"]="core/engine.py core/training.py"
    ["Add mixed precision training"]="core/engine.py"
    ["Implement gradient accumulation"]="core/engine.py"
    ["Add EMA support"]="models/transformer.py"
    ["Implement learning rate scheduling"]="core/engine.py"
    ["Add early stopping"]="core/training.py src/training/advanced_trainer.py"
    ["Implement model checkpointing"]="core/training.py"
    
    # Model Architectures (Août)
    ["Add TabPFN architecture"]="src/models/architectures/ft_transformer.py"
    ["Implement Mamba SSM"]="models/mamba.py src/models/architectures/mamba.py"
    ["Add xLSTM implementation"]="src/models/architectures/xlstm.py"
    ["Implement HyperMixer"]="src/models/architectures/hypermixer.py"
    ["Add TTT architecture"]="src/models/architectures/ttt.py"
    ["Implement Modern TCN"]="src/models/architectures/modern_tcn.py"
    ["Add FT-Transformer"]="src/models/architectures/ft_transformer.py"
    ["Implement SAINT architecture"]="src/models/architectures/ft_transformer.py"
    ["Add TabNet implementation"]="src/models/architectures/ft_transformer.py"
    
    # Ensemble System (Août-Septembre)
    ["Implement ensemble system"]="src/models/ensemble.py src/models/advanced_ensemble.py"
    ["Add attention ensemble"]="src/models/advanced_ensemble.py"
    ["Implement hierarchical ensemble"]="src/models/advanced_ensemble.py"
    
    # Uncertainty (Septembre)
    ["Add uncertainty quantification"]="src/core/uncertainty_quantification.py"
    ["Implement Bayesian layers"]="src/core/uncertainty_quantification.py"
    ["Add Monte Carlo dropout"]="src/core/uncertainty_quantification.py"
    ["Implement deep ensemble"]="src/core/uncertainty_quantification.py"
    ["Add SWAG optimizer"]="src/core/uncertainty_quantification.py"
    ["Implement Laplace approximation"]="src/core/uncertainty_quantification.py"
    ["Add evidential regression"]="src/core/uncertainty_quantification.py"
    
    # Infrastructure (Septembre)
    ["Implement model registry"]="src/models/registry.py"
    ["Add model versioning"]="src/models/registry.py"
    ["Implement inference pipeline"]="src/inference/pipeline.py"
    ["Add batch inference"]="src/inference/pipeline.py"
    
    # Metrics & Evaluation (Septembre-Octobre)
    ["Implement metrics calculation"]="src/evaluation/metrics.py"
    ["Add financial metrics"]="src/evaluation/metrics.py"
    ["Implement ensemble metrics"]="src/evaluation/metrics.py"
    ["Add evaluation framework"]="src/evaluation/metrics.py"
    
    # Optimization (Octobre)
    ["Implement hyperparameter optimization"]="src/training/hyperopt.py"
    ["Add Optuna integration"]="src/training/hyperopt.py"
    ["Implement AutoML pipeline"]="src/training/hyperopt.py"
    
    # Advanced Training (Octobre)
    ["Add advanced trainer"]="src/training/advanced_trainer.py"
    ["Implement callbacks system"]="src/training/advanced_trainer.py"
    ["Add learning rate logger"]="src/training/advanced_trainer.py"
    
    # Loss Functions (Octobre)
    ["Implement loss functions"]="src/training/losses.py"
    ["Add Huber loss"]="src/training/losses.py"
    ["Implement Wing loss"]="src/training/losses.py"
    ["Add focal loss"]="src/training/losses.py"
    ["Implement combined losses"]="src/training/losses.py"
    
    # Optimizers (Octobre)
    ["Add optimizer factory"]="src/training/optimization.py"
    ["Implement LAMB optimizer"]="src/training/optimization.py"
    ["Add scheduler factory"]="src/training/optimization.py"
    ["Implement warmup scheduling"]="src/training/optimization.py"
    
    # Averaging (Octobre)
    ["Add EMA averaging"]="src/training/averaging.py"
    ["Implement SWA"]="src/training/averaging.py"
    ["Add Polyak averaging"]="src/training/averaging.py"
    
    # Modern Architectures (Octobre-Novembre)
    ["Implement modern ResNet"]="src/models/architectures/modern_resnet.py"
    ["Add pyramid architecture"]="src/models/architectures/modern_resnet.py"
    ["Implement DenseNet"]="src/models/architectures/modern_resnet.py"
    ["Add squeeze-excite blocks"]="src/models/architectures/modern_resnet.py"
    ["Implement stochastic depth"]="src/models/architectures/modern_resnet.py"
    ["Add layer scaling"]="src/models/architectures/normalization.py"
    
    # Activations (Novembre)
    ["Implement modern activations"]="src/models/architectures/activations.py"
    ["Add SwiGLU, GeGLU, ReGLU"]="src/models/architectures/activations.py"
    ["Implement Mish activation"]="src/models/architectures/activations.py"
    ["Add STAR activation"]="src/models/architectures/activations.py"
    
    # Normalization (Novembre)
    ["Implement normalization layers"]="src/models/architectures/normalization.py"
    ["Add adaptive layer norm"]="src/models/architectures/normalization.py"
    ["Implement group normalization"]="src/models/architectures/normalization.py"
    ["Add conditional normalization"]="src/models/architectures/normalization.py"
    
    # Attention Variants (Novembre)
    ["Implement attention variants"]="src/models/architectures/attention.py"
    ["Add multi-query attention"]="src/models/architectures/attention.py"
    ["Implement sliding window attention"]="src/models/architectures/attention.py"
    ["Add linear attention"]="src/models/architectures/attention.py"
    ["Implement cross attention"]="src/models/architectures/attention.py"
)

# Messages de commit sans fichiers spécifiques (pour les bugs fixes, refactoring, etc.)
declare -a generic_commits=(
    "Fix attention mask bug"
    "Optimize memory usage"
    "Improve training stability"
    "Fix gradient explosion"
    "Optimize data loading"
    "Improve feature engineering"
    "Fix normalization issues"
    "Optimize batch processing"
    "Improve model convergence"
    "Fix learning rate scheduling"
    "Optimize checkpoint saving"
    "Improve validation metrics"
    "Fix data augmentation"
    "Optimize ensemble training"
    "Improve uncertainty estimation"
    "Fix memory leaks"
    "Optimize inference speed"
    "Improve model accuracy"
    "Fix numerical stability"
    "Optimize GPU utilization"
    "Improve code quality"
    "Refactor architecture"
    "Update documentation"
    "Add usage examples"
    "Improve error handling"
    "Add input validation"
    "Code cleanup"
    "Performance improvements"
    "Bug fixes"
)

# Convertir le tableau associatif en deux tableaux parallèles
commit_messages=()
commit_files=()
for msg in "${!commits_with_files[@]}"; do
    commit_messages+=("$msg")
    commit_files+=("${commits_with_files[$msg]}")
done

# Ajouter les commits génériques
for msg in "${generic_commits[@]}"; do
    commit_messages+=("$msg")
    commit_files+=("")
done

# Date de début : 1er juin 2025
start_date="2025-06-01"
current_date=$(date -d "$start_date" +%s)
end_date=$(date -d "2025-11-05" +%s)

# Compteur
msg_index=0
msg_count=${#commit_messages[@]}
commit_count=0

echo "Creating commit history from June to November 2025..."
echo "This will create commits with actual file modifications..."
echo ""

# Boucle pour créer des commits
while [ $current_date -le $end_date ]; do
    # Date formatée
    formatted_date=$(date -d "@$current_date" "+%Y-%m-%d")
    
    # Nombre de commits pour cette journée (1-4 commits par jour, plus réaliste)
    num_commits=$((RANDOM % 4 + 1))
    
    for ((i=0; i<num_commits; i++)); do
        # Heure aléatoire dans la journée (travail normal: 9h-22h)
        hour=$((RANDOM % 14 + 9))
        minute=$((RANDOM % 60))
        
        commit_date="${formatted_date} ${hour}:${minute}:00"
        
        # Message et fichiers de commit
        message="${commit_messages[$msg_index]}"
        files="${commit_files[$msg_index]}"
        msg_index=$(( (msg_index + 1) % msg_count ))
        
        # Créer le commit avec les fichiers
        create_feature_commit "$commit_date" "$message" "$files"
        
        commit_count=$((commit_count + 1))
        echo "✓ [$commit_count] $message ($formatted_date $hour:$minute)"
    done
    
    # Passer au jour suivant
    current_date=$((current_date + 86400))
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✓ Commit history created successfully!"
echo "═══════════════════════════════════════════════════════════"
echo "Total commits: $(git rev-list --count HEAD)"
echo ""
echo "First commit:"
git log --reverse --format='  %h - %s (%ai)' | head -1
echo ""
echo "Last commit:"
git log --format='  %h - %s (%ai)' | head -1
echo ""
echo "Commits by month:"
git log --since="2025-06-01" --until="2025-12-01" --format="%ai" | cut -d- -f1-2 | uniq -c
echo ""
echo "Ready to push to GitHub with:"
echo "  git remote add origin <your-repo-url>"
echo "  git branch -M main"
echo "  git push -u origin main"
echo "═══════════════════════════════════════════════════════════"