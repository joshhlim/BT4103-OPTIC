import pandas as pd

# 1. Read Excel with selected columns
cols_to_read = [
    'ANAESTHETIST_TEAM', 'ANAESTHETIST_MCR_NO'
]

df = pd.read_excel("Final_Cleaned_Dataset_OPTIC_7.xlsx", usecols=cols_to_read)

# Drop rows with missing values in the relevant columns
df_subset = df[['ANAESTHETIST_TEAM', 'ANAESTHETIST_MCR_NO']].dropna()

# Check unique mappings from TEAM → MCR
team_to_mcr = df_subset.groupby('ANAESTHETIST_TEAM')['ANAESTHETIST_MCR_NO'].nunique()
mcr_to_team = df_subset.groupby('ANAESTHETIST_MCR_NO')['ANAESTHETIST_TEAM'].nunique()

# Print potential issues
print("Teams mapping to multiple MCR numbers:")
print(team_to_mcr[team_to_mcr > 1])

print("\nMCR numbers mapping to multiple Teams:")
print(mcr_to_team[mcr_to_team > 1])

# Teams mapping to multiple MCR numbers:
# Series([], Name: ANAESTHETIST_MCR_NO, dtype: int64)

# MCR numbers mapping to multiple Teams:
# ANAESTHETIST_MCR_NO
# 00766f6a6af7200fc40d48a7b45a479c7769c4f4671847dab76c941f034d44a2     10
# 00fb5ad4c888df0e70d6be622ec527f5279cf19ebfd8a4e36f40f25111c59511    199
# 011e7cd0a425e14eddbe96dad757e8aa6e8a29f638a9d0c78e09a9601025467c    203
# 01f33c236a34b5da1c6aa3e0461e1a46112ec32ca3b8b7237715aca60614e955     43
# 026bed14409abb311734c0db82db8856ce297895f0ac392dea8c270fa6f52ac4     52
#                                                                    ...
# fcf8bab80f39ed2b1b6dc7850c07a22c8d0da7cc31238758d83a975b66a0067d      9
# fdb5995a0016f0a7534d65b64741c366fc965a8a0ffe39ac57d0833318edf828      3
# fee08d94836bf3fe4401cfaf647d9612d21ded3b5d33b694af348b5fd0f96fee    117
# ffbe54ba7c3c7016767fe14aa28c7ba19b562fe14a8d923605fd60cf0a1d7655     12
# ffdda20af1906f8c98d1d8b1cd9ddff8595662aa19b5e3aaa4e3620d298b66e0     61
# Name: ANAESTHETIST_TEAM, Length: 387, dtype: int64
