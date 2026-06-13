import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Désactiver les logs inutiles
tf.get_logger().setLevel('ERROR')

# 1. Paramètres du Problème (Tige Métallique 1D)
ALPHA = 0.05    # Diffusivité thermique
L = 1.0         # Longueur de la tige
T_MAX = 1.0     # Durée de la simulation

# 2. Architecture du PINN
def create_model():
    # Entrée : (x, t)
    inputs = tf.keras.Input(shape=(2,))
    # Couches cachées :
    x = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='glorot_normal')(inputs)
    x = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='glorot_normal')(x)
    # Sortie : Température u
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_model()

# 3. Définition de la Physique et des Pertes
@tf.function
def physics_loss_function(x, t):
    # Calcul des dérivées automatiques
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.stack([x[:, 0], t[:, 0]], axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)    
    # L'équation de la chaleur : u_t - alpha * u_xx = 0
    residual = u_t - ALPHA * u_xx
    return tf.reduce_mean(tf.square(residual))

@tf.function
def compute_loss(X_R, X_IC, U_IC, X_BC, U_BC):
    # 1. Perte Physique (Residual Loss)
    # Séparation des colonnes pour gradient
    x_r = X_R[:, 0:1]
    t_r = X_R[:, 1:2]
    loss_physics = physics_loss_function(x_r, t_r)
    
    # 2. Perte Condition Initiale (t=0)
    u_pred_ic = model(X_IC)
    loss_ic = tf.reduce_mean(tf.square(u_pred_ic - U_IC))
    
    # 3. Perte Conditions aux Limites (x=0 et x=L)
    u_pred_bc = model(X_BC)
    loss_bc = tf.reduce_mean(tf.square(u_pred_bc - U_BC))
    
    # Somme pondérée (on peut ajuster les poids si nécessaire)
    total_loss = loss_physics + loss_ic + loss_bc
    return total_loss, loss_physics, loss_ic, loss_bc

# 4. Génération des Données d'Entraînement
N_R = 2000    # Points de collocation
N_IC = 100    # Points initiaux
N_BC = 100    # Points limites

# A. Points de Collocation (intérieur du domaine)
X_R_np = np.random.uniform(0, L, (N_R, 1))
T_R_np = np.random.uniform(0, T_MAX, (N_R, 1))
X_R = tf.convert_to_tensor(np.hstack((X_R_np, T_R_np)), dtype=tf.float32)

# B. Condition Initiale : u(x,0) = sin(pi*x)
x_ic = np.random.uniform(0, L, (N_IC, 1))
t_ic = np.zeros((N_IC, 1))
X_IC = tf.convert_to_tensor(np.hstack((x_ic, t_ic)), dtype=tf.float32)
U_IC = tf.convert_to_tensor(np.sin(np.pi * x_ic), dtype=tf.float32)

# C. Conditions aux Limites : u(0,t)=0 et u(L,t)=0
t_bc = np.random.uniform(0, T_MAX, (N_BC, 1))
x_bc_0 = np.zeros((N_BC//2, 1))
x_bc_L = np.ones((N_BC//2, 1)) * L
# Combinaison gauche et droite
X_BC_input = np.vstack([np.hstack([x_bc_0, t_bc[:N_BC//2]]), 
                        np.hstack([x_bc_L, t_bc[N_BC//2:]])])
X_BC = tf.convert_to_tensor(X_BC_input, dtype=tf.float32)
U_BC = tf.zeros((N_BC, 1), dtype=tf.float32)

# 5. Entraînement
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
history_loss = []

print("Début de l'entraînement...")
start_time = time.time()

EPOCHS = 3000

for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        loss_val, l_phy, l_ic, l_bc = compute_loss(X_R, X_IC, U_IC, X_BC, U_BC)
    
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Stocker pour le graphique
    history_loss.append(loss_val.numpy())
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss_val.numpy():.6f}")

print(f"Entraînement terminé en {time.time()-start_time:.2f} sec.")

# 6. Solution Exacte (pour comparaison)
def exact_solution(x, t):
    return np.exp(-ALPHA * (np.pi**2) * t) * np.sin(np.pi * x)


# 7. Visualisation Complète

# Création de la grille de test
x_vals = np.linspace(0, L, 100)
t_vals = np.linspace(0, T_MAX, 100)
X_Grid, T_Grid = np.meshgrid(x_vals, t_vals)

# Aplatir pour le modèle
X_test = np.hstack((X_Grid.flatten()[:, None], T_Grid.flatten()[:, None]))
X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)

# Prédiction
u_pred = model.predict(X_test_tf).reshape(X_Grid.shape)
u_exact = exact_solution(X_Grid, T_Grid)
error = np.abs(u_exact - u_pred)

# Configuration de la figure
fig = plt.figure(figsize=(18, 10))

# 1. Heatmap Solution Prédite (t en abscisse, x en ordonnée)
ax1 = fig.add_subplot(2, 2, 1)
# Attention : contourf prend (X, Y, Z). Ici X=Temps, Y=Espace
cp1 = ax1.contourf(T_Grid, X_Grid, u_pred, 100, cmap='jet')
plt.colorbar(cp1, ax=ax1, label='Température u(x,t)')
ax1.set_title("Prédiction PINN : u(x,t)")
ax1.set_xlabel("Temps (t)")
ax1.set_ylabel("Espace (x)")

# 2. Courbe de Loss (Fonction de Coût)
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(history_loss, 'b-', linewidth=2)
ax2.set_yscale('log') # Échelle logarithmique pour mieux voir la convergence
ax2.set_title("Historique de la Loss (Training)")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss (Log scale)")
ax2.grid(True, which="both", ls="-")

# 3. Heatmap de l'Erreur (Accuracy visuelle)
ax3 = fig.add_subplot(2, 2, 3)
cp3 = ax3.contourf(T_Grid, X_Grid, error, 100, cmap='inferno')
plt.colorbar(cp3, ax=ax3, label='Erreur Absolue |u_exact - u_pred|')
ax3.set_title("Carte d'Erreur (Accuracy)")
ax3.set_xlabel("Temps (t)")
ax3.set_ylabel("Espace (x)")

# 4. Coupes (Slices) à différents instants
ax4 = fig.add_subplot(2, 2, 4)
times_to_plot = [0.0, 0.2, 0.5, 0.8]
for t_val in times_to_plot:
    # Trouver l'index le plus proche dans t_vals
    idx = (np.abs(t_vals - t_val)).argmin()
    # Tracer la prédiction
    ax4.plot(x_vals, u_pred[idx, :], '--', linewidth=2, label=f'PINN t={t_val}')
    # Tracer la solution exacte (points discrets)
    ax4.plot(x_vals[::5], u_exact[idx, ::5], 'o', markersize=4, alpha=0.6, label=f'Exact t={t_val}')

ax4.set_title("Comparaison Profils de Température")
ax4.set_xlabel("Espace (x)")
ax4.set_ylabel("Température (u)")
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()
