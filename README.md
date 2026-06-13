# PINN - Équation de la chaleur 1D

Implémentation d'un Physics-Informed Neural Network (PINN) pour résoudre 
l'équation de la chaleur 1D sur une tige métallique.

## Équation résolue

∂u/∂t = α ∂²u/∂x²

avec :
- Condition initiale : u(x,0) = sin(πx)
- Conditions aux limites : u(0,t) = u(L,t) = 0

## Dépendances

```bash
pip install tensorflow numpy matplotlib
```

## Utilisation

```bash
python pinn_heat_v2.py
```

Le script entraîne le réseau pendant 3000 epochs puis affiche :
- La prédiction u(x,t)
- L'historique de la loss
- La carte d'erreur par rapport à la solution exacte
- Des coupes temporelles comparant prédiction et solution exacte
