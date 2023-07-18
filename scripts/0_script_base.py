# --------------------------------------------------------------
# EXTRACAO DAS INFORMACOES DO CONJUNTO DE DADOS DE LINHAS
# DE ONIBUS DE FLORIANOPOLIS 
# --------------------------------------------------------------
import pandas as pd
import numpy as np
import glob
import time

import matplotlib.pyplot as plt

### CONSTANTES
ticks = time.time()
pd.set_option('display.max_columns', None)
CAMINHO_DIRETORIO = "dataset/"
PATTERN = "*.txt"
# - data e hora de partida
# - data e hora de chegada
# - sentido (ida, volta)
# - linha
# - código do Veículo
# - total de giros => número de passageiros que passaram na catraca dos ônibus
# - quilômetros percorridos na viagem 
# - duração da viagem

def ler_diretorio_em_dataframe(caminho: str, delimitador=None, colunas=None) -> list:
        ## Funcao responsavel por ler dados de um diretorio
        ## Recebe como parametros
        ## param caminho: caminho do diretorio para leitura
        delimitador = ";" if not delimitador else delimitador
        colunas = [] if not colunas else colunas
        return [
            pd.read_csv(nome_arquivo, delimiter=delimitador, low_memory=False, names=colunas)
            for nome_arquivo in glob.glob(caminho)
        ]

def concatenar_dataframes(lista_dataframes):
    ## Concatena lista de dataframes em um unico
    ## Retorna dataframe concatenado
    return pd.concat(lista_dataframes)

def executar_script():
    # Script que executará o restante do programa
    # Declaracao de classes
    # Lendo diretorios 
    # Colunas
    n = [
        "DataPartida",
        "HoraPartida",
        "DataChegada",
        "HoraChegada",
        "Sentido",
        "Linha",
        "CodigoVeiculo",
        "DuracaoViagem",
        "GirosCatraca",
        "KmPercorridos"
    ]
    dataframes = ler_diretorio_em_dataframe("{0}/{1}".format(CAMINHO_DIRETORIO, PATTERN), colunas=n)
    # Resultado concatenado
    resultado_concatenado = concatenar_dataframes(dataframes)
    # Retornando resultado concatenado
    return resultado_concatenado

dados_brutos = executar_script()

# 1. Analisar os dados do dataframe
# 2. Unindo data e hora partida em mesma coluna e data e hora chegada em mesma coluna
dados_analises = dados_brutos.copy()

def concatenar_data_e_hora(conteudo, coluna_data, coluna_hora):
    """
    Funcao que busca concatenar da hora retornando um datetime 
    """
    conteudo[coluna_data] = conteudo[coluna_data].apply(lambda x: x[:10].replace(" 0","").strip())
    conteudo[coluna_hora] = conteudo[coluna_hora].apply(lambda x: x[-8:].strip())
    return pd.to_datetime(conteudo[coluna_data] + " " + conteudo[coluna_hora], dayfirst=True)

def converter_tipo_coluna(conteudo, nome_da_coluna, tipo_desejado):
    """
    Funcao que converte o tipo da coluna do dado dataframe
    """
    if tipo_desejado == 'float':
       conteudo[nome_da_coluna] = conteudo[nome_da_coluna].apply(lambda x: x.replace(",","."))
       conteudo[nome_da_coluna] = conteudo[nome_da_coluna].astype(tipo_desejado)
    elif tipo_desejado in ['minutesOnly']:
       conteudo[nome_da_coluna] = pd.to_datetime(conteudo[nome_da_coluna], dayfirst=True).dt.minute
    elif tipo_desejado in ['dateOnly']:
       conteudo[nome_da_coluna] = pd.to_datetime(conteudo[nome_da_coluna], dayfirst=True).dt.date
    elif tipo_desejado in ['timeOnly']:
       conteudo[nome_da_coluna] = pd.to_datetime(conteudo[nome_da_coluna], dayfirst=True).dt.time
    else:
       conteudo[nome_da_coluna] = conteudo[nome_da_coluna].astype(tipo_desejado) 
    
    return conteudo

def gerar_tick_de_um_datetime(conteudo, nome_coluna_datetime):
    """
    Funcao que converte uma coluna de datetime gerando um tick
    """
    return conteudo[nome_coluna_datetime].apply(
        lambda x: time.mktime(x.timetuple())
    )

dados_analises['DataHoraPartida'] = concatenar_data_e_hora(dados_analises.copy(), 'DataPartida', 'HoraPartida')
dados_analises['DataHoraChegada'] = concatenar_data_e_hora(dados_analises.copy(), 'DataChegada', 'HoraChegada')
dados_analises['TicksPartida'] = gerar_tick_de_um_datetime(dados_analises, 'DataHoraPartida')
dados_analises['TicksChegada'] = gerar_tick_de_um_datetime(dados_analises, 'DataHoraChegada')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'GirosCatraca', 'int')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'KmPercorridos', 'float')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'DuracaoViagem', 'minutesOnly')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'DataPartida', 'dateOnly')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'HoraPartida', 'timeOnly')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'DataChegada', 'dateOnly')
dados_analises = converter_tipo_coluna(dados_analises.copy(), 'HoraChegada', 'timeOnly')

# 3. Limpeza de dados 1: Retirar colunas onde a datahora partida e datahora chegada são iguais
def filtrar_informacao(conteudo, predicado):
    """
    Funcao que realizar um busca no dataframe retornando seu resultado
    """
    return conteudo.query(predicado).reset_index().drop(columns=["index"])

dados_primeiro_filtro = filtrar_informacao(dados_analises.copy(), "DataHoraPartida < DataHoraChegada")
# 4. Limpeza de dados 2, retirando dados onde quilometros percorridos é zero e catraca maior que 0.
dados_segundo_filtro = filtrar_informacao(dados_primeiro_filtro.copy(), "~(KmPercorridos == 0.00 & GirosCatraca > 0) & ~(KmPercorridos == 0.00 & GirosCatraca == 0)") 
# dados_segundo_filtro[['GirosCatraca','KmPercorridos']].hist(bins=20, range=[2.5, 75.5])
# dados_segundo_filtro[['GirosCatraca','KmPercorridos', 'DuracaoViagem']]
# 5. Limpeza de dados 3, retirando dados onde velocidade media ultrapasse os 80km/h
dados_terceiro_filtro = filtrar_informacao(dados_segundo_filtro.copy(), "KmPercorridos / (DuracaoViagem/60) <= 80")
dados_terceiro_filtro['Dur'] = ((dados_terceiro_filtro['DataHoraChegada'] - dados_terceiro_filtro['DataHoraPartida']).dt.total_seconds()//60).astype('int')
# 6. Limpeza de dados 4, retirando duracoes erradas de viagem
dados_quarto_filtro = filtrar_informacao(dados_terceiro_filtro.copy(), "Dur == DuracaoViagem")
# 7. Limpeza de dados 5, retirando ocorrencias em que um veiculo esta em duas viagens simultaneas
dados_quinto_filtro = dados_quarto_filtro.drop_duplicates(subset=["CodigoVeiculo", "TicksPartida"])
# 8. Análise, ordernar pelo tempo 
dados_ordernados_por_tick_linha = dados_quinto_filtro.copy().sort_values(by=["Linha", "TicksPartida", "Sentido"])
# 9. Análise, ida e volta de uma linha
dados_ida = filtrar_informacao(dados_ordernados_por_tick_linha.copy(), "Sentido == 'Ida'")
dados_ida = dados_ida.rename(
    columns=dict(
        zip(
            list(dados_ida.columns), 
            ["ida_" + i for i in list(dados_ida.columns)]
        )
    )
)
dados_volta = filtrar_informacao(dados_ordernados_por_tick_linha.copy(), "Sentido == 'Volta'")
dados_volta = dados_volta.rename(
    columns=dict(
        zip(
            list(dados_volta.columns),
            ["volta_" + i for i in list(dados_volta.columns)]
        )
    )
)
# 10. Análise, exemplo de linha que não possui ida
dados_ida_por_linha = filtrar_informacao(dados_ida.copy(), "ida_Linha == '100'")
# 11. Análise, exemplo de linha que apenas possui volta
dados_volta_por_linha = filtrar_informacao(dados_volta.copy(), "volta_Linha == '100'")
# 12. Análise, avaliando capacidade total dos veiculos
capacidade_total_veiculos = dados_ordernados_por_tick_linha.groupby("Linha")["GirosCatraca"].agg('max')
# 13. Analise, adicionando colunas de dia da semana
dados_ordernados_por_tick_linha["dia_semana_partida"] = dados_ordernados_por_tick_linha.copy()["DataHoraPartida"].dt.day_name()
dados_ordernados_por_tick_linha["dia_semana_chegada"] = dados_ordernados_por_tick_linha.copy()["DataHoraChegada"].dt.day_name()
dados_com_semana = dados_ordernados_por_tick_linha.copy()
# 14. Analise, filtrando 
import pandas as pd
pd.set_option('display.max_rows', 20)
dados_com_semana["DataHoraPartida120"] = pd.to_datetime(dados_com_semana["DataHoraPartida"]).dt.floor('1440T')
dados_com_semana_grp = dados_com_semana.copy().groupby(["Linha","DataHoraPartida120", "dia_semana_partida", "CodigoVeiculo"], as_index=False).agg({"GirosCatraca": ["sum"], "Sentido": ["count"], "Dur": ["sum"], "KmPercorridos": ["sum"]})
dados_com_semana_grp.columns = [i.replace("sum","").replace("count","") for i in list(map(''.join, dados_com_semana_grp.columns.values))]
dados_com_semana_grp = dados_com_semana_grp.rename(columns={"Sentido": "Viagens", "Dur": "Duracao"})
dados_com_semana_grp["VelocidadeMedia"] = (dados_com_semana_grp["KmPercorridos"]/(dados_com_semana_grp["Duracao"]/60))
dados_filtrados_com_linha_100 = dados_com_semana_grp.copy()[dados_com_semana_grp['Linha'] == '100']
# Salvando em content/
dados_filtrados_com_linha_100.to_csv('content/dataset.csv')
