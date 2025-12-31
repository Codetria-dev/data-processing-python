"""
Sistema de Processamento de Dados Governamentais
Processa dados de entrada, realiza análises estatísticas e gera relatórios.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def formatar_tabela(df, espacamento_colunas=3):
    """
    Formata DataFrame para exibição com espaçamento adequado.
    
    Args:
        df (pd.DataFrame): DataFrame a ser formatado
        espacamento_colunas (int): Número de espaços entre colunas
        
    Returns:
        str: String formatada da tabela
    """
    if df.empty:
        return ""
    
    # Converter DataFrame para string com espaçamento
    output = []
    colunas = df.columns.tolist()
    
    # Calcular largura de cada coluna
    larguras = {}
    for col in colunas:
        larguras[col] = max(
            len(str(col)),
            df[col].astype(str).str.len().max() if len(df) > 0 else 0
        )
    
    # Cabeçalho
    header = "  ".join([str(col).ljust(larguras[col] + espacamento_colunas) for col in colunas])
    output.append(header)
    output.append("-" * len(header))
    
    # Linhas de dados
    for idx, row in df.iterrows():
        linha = "  ".join([
            str(row[col]).ljust(larguras[col] + espacamento_colunas) 
            for col in colunas
        ])
        output.append(linha)
    
    return "\n".join(output)


def processar_dados(arquivo_entrada, arquivo_saida):
    """
    Processa dados de entrada, realiza limpeza e transformações.
    
    Args:
        arquivo_entrada (str): Caminho do arquivo CSV de entrada
        arquivo_saida (str): Caminho do arquivo CSV de saída
        
    Returns:
        pd.DataFrame: DataFrame com dados processados ou None em caso de erro
    """
    print()
    print("=" * 80)
    print(" " * 25 + "PROCESSAMENTO DE DADOS DO GOVERNO")
    print(" " * 20 + "Base de Dados Publicos")
    print("=" * 80)
    print()

    # 1. Ler dados de entrada
    print(" " * 5 + "[1/8] Lendo dados de entrada...")
    print()
    try:
        df = pd.read_csv(arquivo_entrada, sep=';', thousands=',')
        print(" " * 10 + "[OK] Dados lidos com sucesso")
        print(" " * 10 + f"     Total de linhas: {len(df):,}")
        print(" " * 10 + f"     Dimensoes: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print(" " * 10 + f"     Colunas: {', '.join(df.columns.tolist())}")
        print()
    except FileNotFoundError:
        print(" " * 10 + f"[ERRO] Arquivo nao encontrado: {arquivo_entrada}")
        return None
    except Exception as e:
        print(" " * 10 + f"[ERRO] Erro ao ler arquivo: {e}")
        return None

    # 2. Tratar nomes das colunas
    print(" " * 5 + "[2/8] Tratando nomes das colunas...")
    print()
    colunas_antes = df.columns.tolist()
    df.columns = df.columns.str.strip().str.lower()
    print(" " * 10 + f"[OK] Colunas padronizadas")
    print(" " * 10 + f"     Colunas apos tratamento: {', '.join(df.columns.tolist())}")
    print()

    # 3. Tratar valores nulos
    print(" " * 5 + "[3/8] Verificando valores nulos...")
    print()
    nulos_antes = df.isnull().sum().sum()
    print(" " * 10 + f"[INFO] Valores nulos encontrados: {nulos_antes:,}")

    if nulos_antes > 0:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        for coluna in colunas_numericas:
            media = df[coluna].mean()
            if not pd.isna(media):
                df[coluna].fillna(media, inplace=True)
        print(" " * 10 + f"[OK] Valores nulos preenchidos com a media das colunas")
    else:
        print(" " * 10 + f"[OK] Nenhum valor nulo encontrado")
    print()

    # 4. Remover duplicatas
    print(" " * 5 + "[4/8] Verificando duplicatas...")
    print()
    duplicatas_antes = df.duplicated().sum()
    print(" " * 10 + f"[INFO] Linhas duplicadas encontradas: {duplicatas_antes:,}")

    if duplicatas_antes > 0:
        df.drop_duplicates(inplace=True)
        print(" " * 10 + f"[OK] Duplicatas removidas")
        print(" " * 10 + f"     Linhas restantes: {len(df):,}")
    else:
        print(" " * 10 + f"[OK] Nenhuma duplicata encontrada")
    print()

    # 5. Padronizar dados
    print(" " * 5 + "[5/8] Padronizando dados...")
    print()

    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y', errors='coerce')
        datas_validas = df['data'].notna().sum()
        print(" " * 10 + f"[OK] Coluna 'data' convertida para datetime")
        print(" " * 10 + f"     Datas validas: {datas_validas:,} de {len(df):,}")
        print()

    if 'valor' in df.columns:
        df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
        valores_validos = df['valor'].notna().sum()
        print(" " * 10 + f"[OK] Coluna 'valor' convertida para numerico")
        print(" " * 10 + f"     Valores validos: {valores_validos:,} de {len(df):,}")
        print()

        if 'data' in df.columns and df['data'].notna().any():
            df['ano'] = df['data'].dt.year
            df['mes'] = df['data'].dt.month
            df['mes_nome'] = df['data'].dt.strftime('%b')
            df['mes_ano'] = df['data'].dt.strftime('%Y-%m')

            data_min = df['data'].min()
            data_max = df['data'].max()
            if pd.notna(data_min) and pd.notna(data_max):
                print(" " * 10 + f"[INFO] Periodo dos dados: {data_min.date()} a {data_max.date()}")
                print()

    # 6. Análise estatística
    print(" " * 5 + "[6/8] Realizando analise estatistica...")
    print()

    if 'valor' in df.columns and df['valor'].notna().any():
        estatisticas = {
            'Total de Registros': len(df),
            'Periodo Total': f"{df['ano'].min()} a {df['ano'].max()}" if 'ano' in df.columns else 'N/A',
            'Total de Anos': df['ano'].nunique() if 'ano' in df.columns else 0,
            'Media Geral': df['valor'].mean(),
            'Mediana': df['valor'].median(),
            'Desvio Padrao': df['valor'].std(),
            'Valor Minimo': df['valor'].min(),
            'Valor Maximo': df['valor'].max(),
            'Valores Negativos': (df['valor'] < 0).sum(),
            'Soma Total': df['valor'].sum()
        }

        print(" " * 10 + "ESTATISTICAS GERAIS:")
        print(" " * 10 + "-" * 60)
        print()
        for chave, valor in estatisticas.items():
            if isinstance(valor, (int, np.integer)):
                print(f" " * 15 + f"{chave:.<35} {valor:>20,}")
            elif isinstance(valor, float):
                print(f" " * 15 + f"{chave:.<35} {valor:>20,.2f}")
            else:
                print(f" " * 15 + f"{chave:.<35} {valor:>20}")
        print()

        if 'ano' in df.columns:
            print(" " * 10 + "ESTATISTICAS POR ANO:")
            print(" " * 10 + "-" * 60)
            print()
            estatisticas_ano = df.groupby('ano')['valor'].agg([
                ('Registros', 'count'),
                ('Media', 'mean'),
                ('Soma', 'sum'),
                ('Minimo', 'min'),
                ('Maximo', 'max')
            ]).round(2)
            
            # Formatar tabela com espaçamento
            tabela_formatada = formatar_tabela(estatisticas_ano, espacamento_colunas=5)
            for linha in tabela_formatada.split('\n'):
                print(" " * 15 + linha)
            print()

        if 'mes_nome' in df.columns:
            print(" " * 10 + "ESTATISTICAS POR MES:")
            print(" " * 10 + "-" * 60)
            print()
            estatisticas_mes = df.groupby('mes_nome')['valor'].agg([
                ('Registros', 'count'),
                ('Media', 'mean'),
                ('Soma', 'sum')
            ]).round(2).sort_index()
            
            tabela_formatada = formatar_tabela(estatisticas_mes, espacamento_colunas=5)
            for linha in tabela_formatada.split('\n'):
                print(" " * 15 + linha)
            print()

    # 7. Calcular tendências
    print(" " * 5 + "[7/8] Calculando tendencias...")
    print()

    if 'valor' in df.columns and 'ano' in df.columns:
        anos_unicos = sorted(df['ano'].dropna().unique())
        if len(anos_unicos) >= 2:
            primeiro_ano = anos_unicos[0]
            ultimo_ano = anos_unicos[-1]

            soma_primeiro = df[df['ano'] == primeiro_ano]['valor'].sum()
            soma_ultimo = df[df['ano'] == ultimo_ano]['valor'].sum()

            if soma_primeiro > 0:
                crescimento_total = ((soma_ultimo - soma_primeiro) / soma_primeiro) * 100
                print(" " * 10 + "CRESCIMENTO TOTAL:")
                print(" " * 10 + "-" * 60)
                print()
                print(" " * 15 + f"Periodo: {primeiro_ano} -> {ultimo_ano}")
                print(" " * 15 + f"Taxa de crescimento: {crescimento_total:.2f}%")
                print()

            print(" " * 10 + "CRESCIMENTO ANO A ANO:")
            print(" " * 10 + "-" * 60)
            print()
            soma_por_ano = df.groupby('ano')['valor'].sum()
            for i in range(1, len(soma_por_ano)):
                ano_atual = soma_por_ano.index[i]
                ano_anterior = soma_por_ano.index[i-1]
                valor_atual = soma_por_ano.iloc[i]
                valor_anterior = soma_por_ano.iloc[i-1]
                if valor_anterior > 0:
                    crescimento = ((valor_atual - valor_anterior) / valor_anterior) * 100
                    print(" " * 15 + f"{ano_anterior} -> {ano_atual}: {crescimento:>10.2f}%")
            print()

        if 'ano' in df.columns:
            top_anos = df.groupby('ano')['valor'].sum().nlargest(5)
            print(" " * 10 + "TOP 5 ANOS COM MAIORES VALORES:")
            print(" " * 10 + "-" * 60)
            print()
            for posicao, (ano, valor) in enumerate(top_anos.items(), 1):
                print(" " * 15 + f"{posicao}o. {ano}: {valor:>20,.2f}")
            print()

        if 'mes_nome' in df.columns:
            top_meses = df.groupby('mes_nome')['valor'].mean().nlargest(3)
            print(" " * 10 + "TOP 3 MESES COM MAIOR MEDIA:")
            print(" " * 10 + "-" * 60)
            print()
            for posicao, (mes, valor) in enumerate(top_meses.items(), 1):
                print(" " * 15 + f"{posicao}o. {mes}: {valor:>20,.2f}")
            print()

    # 8. Salvar arquivo processado
    print(" " * 5 + f"[8/8] Salvando dados processados...")
    print()
    try:
        df = df.sort_values('data') if 'data' in df.columns else df

        # Salvar CSV com espaçamento melhorado
        df.to_csv(arquivo_saida, index=False, sep=';', decimal='.', encoding='utf-8')
        print(" " * 10 + f"[OK] Arquivo CSV salvo: {arquivo_saida}")
        print()

        arquivo_excel = arquivo_saida.replace('.csv', '.xlsx')
        df.to_excel(arquivo_excel, index=False)
        print(" " * 10 + f"[OK] Arquivo Excel salvo: {arquivo_excel}")
        print()
    except Exception as e:
        print(" " * 10 + f"[ERRO] Erro ao salvar arquivo: {e}")
        print()
        return None

    print("=" * 80)
    print(" " * 25 + "PROCESSAMENTO CONCLUIDO COM SUCESSO")
    print("=" * 80)
    print()

    return df


def gerar_relatorio_completo(df, arquivo_saida):
    """
    Gera um relatório textual completo com todas as análises realizadas.
    
    Args:
        df (pd.DataFrame): DataFrame com dados processados
        arquivo_saida (str): Caminho base para salvar o relatório
        
    Returns:
        str: Conteúdo do relatório gerado
    """
    if df is None or df.empty:
        print(" " * 10 + "[ERRO] DataFrame vazio ou invalido para gerar relatorio")
        return None

    relatorio = f"""
{'=' * 80}
{' ' * 25}RELATORIO DE ANALISE - DADOS DO GOVERNO
{'=' * 80}


INFORMACOES GERAIS:
{'-' * 80}
Total de Registros: {len(df):,}
Periodo Analisado: {df['data'].min().date()} a {df['data'].max().date()}
Total de Anos: {df['ano'].nunique()}
Total de Meses Unicos: {df['mes_ano'].nunique()}


ESTATISTICAS DOS VALORES:
{'-' * 80}
Media Geral: {df['valor'].mean():,.2f}
Mediana: {df['valor'].median():,.2f}
Desvio Padrao: {df['valor'].std():,.2f}
Valor Minimo: {df['valor'].min():,.2f}
Valor Maximo: {df['valor'].max():,.2f}
Soma Total: {df['valor'].sum():,.2f}
Valores Negativos: {(df['valor'] < 0).sum():,} registros


DISTRIBUICAO POR ANO:
{'-' * 80}
"""

    soma_por_ano = df.groupby('ano')['valor'].sum()
    for ano, soma in soma_por_ano.items():
        relatorio += f"{ano}: {soma:,.2f}\n"
    relatorio += "\n"

    relatorio += f"""
TENDENCIAS E ANALISE:
{'-' * 80}
"""

    anos_unicos = sorted(df['ano'].unique())
    if len(anos_unicos) >= 2:
        primeiro_ano = anos_unicos[0]
        ultimo_ano = anos_unicos[-1]
        soma_primeiro = df[df['ano'] == primeiro_ano]['valor'].sum()
        soma_ultimo = df[df['ano'] == ultimo_ano]['valor'].sum()

        if soma_primeiro > 0:
            crescimento = ((soma_ultimo - soma_primeiro) / soma_primeiro) * 100
            relatorio += f"Crescimento Total ({primeiro_ano} -> {ultimo_ano}): {crescimento:.2f}%\n"

    top_anos = df.groupby('ano')['valor'].sum().nlargest(3)
    relatorio += f"\nTop 3 Anos com Maiores Valores:\n"
    for pos, (ano, valor) in enumerate(top_anos.items(), 1):
        relatorio += f"  {pos}o. {ano}: {valor:,.2f}\n"

    relatorio += f"""

DISTRIBUICAO POR MES:
{'-' * 80}
"""
    media_por_mes = df.groupby('mes_nome')['valor'].mean()
    meses_ordem = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for mes in meses_ordem:
        if mes in media_por_mes.index:
            relatorio += f"{mes}: {media_por_mes[mes]:,.2f}\n"

    relatorio += f"""

ANALISE POR DECADA:
{'-' * 80}
"""
    df['decada'] = (df['ano'] // 10) * 10
    estatisticas_decada = df.groupby('decada')['valor'].agg([
        ('Registros', 'count'),
        ('Media', 'mean'),
        ('Soma', 'sum')
    ]).round(2)

    for decada, stats in estatisticas_decada.iterrows():
        relatorio += f"Decada {decada}s:\n"
        relatorio += f"  - Registros: {stats['Registros']:,}\n"
        relatorio += f"  - Media: {stats['Media']:,.2f}\n"
        relatorio += f"  - Soma: {stats['Soma']:,.2f}\n"
        relatorio += "\n"

    relatorio += f"""
RESUMO DE QUALIDADE DE DADOS:
{'-' * 80}
Dados Validos: {df['valor'].notna().sum():,} de {len(df):,}
Dados Faltantes: {df['valor'].isna().sum():,}
Valores Unicos: {df['valor'].nunique():,}
Intervalo de Datas: {df['data'].max() - df['data'].min()}


{'=' * 80}
RELATORIO GERADO EM: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
{'=' * 80}
"""

    nome_relatorio = arquivo_saida.replace('.csv', '_relatorio.txt')
    try:
        with open(nome_relatorio, 'w', encoding='utf-8') as f:
            f.write(relatorio)
        print(" " * 10 + f"[OK] Relatorio salvo em: {nome_relatorio}")
        print()
    except Exception as e:
        print(" " * 10 + f"[ERRO] Erro ao salvar relatorio: {e}")
        print()

    return relatorio


def exportar_estatisticas_detalhadas(df, arquivo_saida):
    """
    Exporta estatísticas detalhadas para arquivos CSV separados.
    
    Args:
        df (pd.DataFrame): DataFrame com dados processados
        arquivo_saida (str): Caminho base para salvar os arquivos
        
    Returns:
        tuple: Tupla com DataFrames de estatísticas (por ano, por mês)
    """
    if df is None or df.empty:
        print(" " * 10 + "[ERRO] DataFrame vazio ou invalido para exportar estatisticas")
        return None, None

    try:
        stats_ano = df.groupby('ano').agg(
            registros=('valor', 'count'),
            soma=('valor', 'sum'),
            media=('valor', 'mean'),
            mediana=('valor', 'median'),
            desvio_padrao=('valor', 'std'),
            minimo=('valor', 'min'),
            maximo=('valor', 'max')
        ).round(2)

        stats_mes = df.groupby('mes_nome').agg(
            registros=('valor', 'count'),
            soma=('valor', 'sum'),
            media=('valor', 'mean'),
            mediana=('valor', 'median')
        ).round(2)

        arquivo_stats_ano = arquivo_saida.replace('.csv', '_estatisticas_ano.csv')
        arquivo_stats_mes = arquivo_saida.replace('.csv', '_estatisticas_mes.csv')

        stats_ano.to_csv(arquivo_stats_ano, sep=';', decimal='.')
        print(" " * 10 + f"[OK] Estatisticas por ano salvas em: {arquivo_stats_ano}")
        print()

        stats_mes.to_csv(arquivo_stats_mes, sep=';', decimal='.')
        print(" " * 10 + f"[OK] Estatisticas por mes salvas em: {arquivo_stats_mes}")
        print()

        return stats_ano, stats_mes

    except Exception as e:
        print(" " * 10 + f"[ERRO] Erro ao exportar estatisticas: {e}")
        print()
        return None, None


def main():
    """
    Função principal que executa o processamento completo dos dados.
    """
    # Configurações - Todos os arquivos serão salvos na pasta data
    PASTA_DADOS = "data"
    ARQUIVO_ENTRADA = os.path.join(PASTA_DADOS, "governo.csv")
    NOME_ARQUIVO_SAIDA = "governo_processado.csv"
    ARQUIVO_SAIDA = os.path.join(PASTA_DADOS, NOME_ARQUIVO_SAIDA)

    print()
    print("=" * 80)
    print(" " * 25 + "SISTEMA DE PROCESSAMENTO DE DADOS")
    print(" " * 20 + "Dados Governamentais - Base Publica")
    print("=" * 80)
    print()

    # Processar dados
    dados_processados = processar_dados(ARQUIVO_ENTRADA, ARQUIVO_SAIDA)

    if dados_processados is not None:
        # Gerar relatório completo
        print(" " * 5 + "Gerando relatorio completo...")
        print()
        relatorio = gerar_relatorio_completo(dados_processados, ARQUIVO_SAIDA)

        # Exportar estatísticas detalhadas
        print(" " * 5 + "Exportando estatisticas detalhadas...")
        print()
        stats_ano, stats_mes = exportar_estatisticas_detalhadas(dados_processados, ARQUIVO_SAIDA)

        # Resumo final
        print("=" * 80)
        print(" " * 30 + "RESUMO DA EXECUCAO")
        print("=" * 80)
        print()

        print(" " * 5 + "ARQUIVOS GERADOS:")
        print(" " * 5 + "-" * 70)
        print()
        print(" " * 10 + f"1. {ARQUIVO_SAIDA}")
        print(" " * 15 + "   Dados processados completos")
        print()
        print(" " * 10 + f"2. {ARQUIVO_SAIDA.replace('.csv', '.xlsx')}")
        print(" " * 15 + "   Versao Excel dos dados")
        print()
        print(" " * 10 + f"3. {ARQUIVO_SAIDA.replace('.csv', '_relatorio.txt')}")
        print(" " * 15 + "   Relatorio de analise textual")
        print()
        print(" " * 10 + f"4. {ARQUIVO_SAIDA.replace('.csv', '_estatisticas_ano.csv')}")
        print(" " * 15 + "   Estatisticas agregadas por ano")
        print()
        print(" " * 10 + f"5. {ARQUIVO_SAIDA.replace('.csv', '_estatisticas_mes.csv')}")
        print(" " * 15 + "   Estatisticas agregadas por mes")
        print()

        print(" " * 5 + "DADOS ESTATISTICOS:")
        print(" " * 5 + "-" * 70)
        print()
        print(" " * 10 + dados_processados['valor'].describe().to_string().replace('\n', '\n' + ' ' * 10))
        print()

        print(" " * 5 + "PERIODO PROCESSADO:")
        print(" " * 5 + "-" * 70)
        print()
        data_min = dados_processados['data'].min().date()
        data_max = dados_processados['data'].max().date()
        dias_totais = (dados_processados['data'].max() - dados_processados['data'].min()).days
        print(" " * 10 + f"Data inicial: {data_min}")
        print(" " * 10 + f"Data final:   {data_max}")
        print(" " * 10 + f"Total de dias: {dias_totais:,}")
        print()

        print("=" * 80)
        print(" " * 25 + "PROCESSAMENTO FINALIZADO COM SUCESSO")
        print("=" * 80)
        print()
    else:
        print("=" * 80)
        print(" " * 25 + "PROCESSAMENTO INTERROMPIDO")
        print(" " * 20 + "Verifique os erros acima e tente novamente")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()