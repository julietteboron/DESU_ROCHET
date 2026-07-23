#imputation

from sklearn.impute import SimpleImputer

raw_df_restr_8 = df_restr_8.copy() ##je garde une copie du dataframe avant imputation pour pouvoir comparer les résultats après imputation.


# On sépare les features (à imputer) de la cible
features_restantes = df_restr_8.columns.drop('nutriscore_grade')

imputer = SimpleImputer(strategy='mean')

df_restr_8[features_restantes] = imputer.fit_transform(df_restr_8[features_restantes])

print(df_restr_8.isnull().sum())  # vérification : tout devrait être à 0
df_restr_8.head()

#comparer avant et après imputation pour voir si les valeurs imputées sont cohérentes avec les autres valeurs de la colonne.

def compare_dist(feature, raw_df_restr_8, df_restr_8):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    
    # Bins communs calculés sur la plage combinée des deux distributions
    combined_min = min(raw_df_restr_8[feature].min(), df_restr_8[feature].min())
    combined_max = max(raw_df_restr_8[feature].max(), df_restr_8[feature].max())
    bins = np.linspace(combined_min, combined_max, 30)  # 30 bins, ajustable
    
    ax = axes[0]
    sns.histplot(raw_df_restr_8.loc[:, feature], kde=True, ax=ax, bins=bins)
    ax.set_title(f"Raw {feature}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax = axes[1]
    sns.histplot(df_restr_8.loc[:, feature], kde=True, ax=ax, bins=bins)
    ax.set_title(f"Imputed {feature}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# Appel pour chaque feature (boucle sur toutes tes colonnes numériques)
for feature in features_restantes:
    compare_dist(feature, raw_df_restr_8, df_restr_8)