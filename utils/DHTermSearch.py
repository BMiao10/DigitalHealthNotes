import numpy as np
import pandas as pd
import requests
import json
import bs4

def query_ctgov_api(query, return_fields, verbose=False, n_lim=None, 
                           study_url = "https://ClinicalTrials.gov/api/query/study_fields?", search_field=None):
    """
    Retrieve data from ClinicalTrials.gov
    Search ClinicalTrials.gov data using requests
    
    Params:
        query (str): text query
        return_fields (list<str>): fields to return, see 
        n_lim (int, None): limit to number of queries to return
        study_url (str): Clinical trials api link
        search_field (list<str>): limit ClinicalTrials.gov fields to search for query in
        // verbose (bool): print query numbers
        
    """

    # search for clinical trials and get data
    if len(query)==0: 
        return

    if len(return_fields) > 20:
        return("Too many search results to return! Max 20")

    
    # first search
    clinical_df = pd.DataFrame()
    curr_min = 1
    curr_max = 1000

    # this is a little confusing but
    # "fields"=which fields to return results for ="return_fields"
    # "field"=field to search for query in = "search_field"
    params = {"expr":query,
                "fmt":"JSON", 
                "fields" : ",".join(return_fields),
                "min_rnk":curr_min,
                "max_rnk":curr_max
            }
    if search_field is not None: params["field"] = search_field

    r = requests.get(study_url, params)
    #if len(r.text) < 1000: st.write(r.text) # QA 
    info = r.json()

    # get total number of studies
    if info["StudyFieldsResponse"]["NStudiesReturned"]>0:
        n_total = info["StudyFieldsResponse"]["NStudiesFound"]
        info = info['StudyFieldsResponse']["StudyFields"]
        clinical_df = pd.DataFrame(info)
    else:
        
        return "No studies found"
    
    if n_lim is not None: 
        n_total = min(n_total, n_lim)

    if n_total > curr_max:
        while n_total > curr_max:
            curr_min = curr_min + 999
            curr_max = min(n_total, curr_max + 999)

            # get new data
            params = {"expr":query,
                "fmt":"JSON", 
                "fields" : ",".join(return_fields),
                "min_rnk":curr_min,
                "max_rnk":curr_max
            }
            if search_field is not None: params["field"] = search_field

            r = requests.get(study_url, params)
            info = r.json()

            # append to dataframe
            if info["StudyFieldsResponse"]["NStudiesReturned"]>0:
                info = info['StudyFieldsResponse']["StudyFields"]
                curr_df = pd.DataFrame(info)
                clinical_df = pd.concat([clinical_df, curr_df])

            #if verbose: st.write("%d/%d records retrieved"%(curr_max,n_total))
    #else:
        #if verbose:  st.write("%d records retrieved"%n_total)

    # clean up values
    del clinical_df["Rank"]
    clinical_df = clinical_df.replace(np.nan, "")
    clinical_df = clinical_df.replace("\t", ' ')
    clinical_df = clinical_df.replace("\n", ' ')
    clinical_df = clinical_df.applymap(lambda x: "\t".join(x) if type(x)==list else x)
    clinical_df = clinical_df.replace(np.nan, "")

    # drop duplicates
    #clinical_df = clinical_df.set_index("NCTId")
    clinical_df = clinical_df.drop_duplicates()

    return clinical_df

def get_pmc_ids(query, search_params):
    """
    Uses eutils to retrieve PMC abstracts from a search
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Perform PMC search
    search_payload = {
        "db": "pmc",
        "term": query,
        "retmode":"json",
        "retmax": 9999
    }
    search_payload.update(search_params)
    
    search_response = requests.get(base_url+"esearch.fcgi", params=search_payload)
    search_data = search_response.json()
    
    return search_data

def fetch_full_pmc_text(uid_list):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = base_url + "efetch.fcgi"

    # Fetch the full text of articles using the UID list
    fetch_payload = {
        "db": "pmc",
        "id": ",".join(uid_list),
        "retmode": "xml",
        "rettype": "full"
    }
    fetch_response = requests.get(fetch_url, params=fetch_payload)

    # Convert XML response to JSON
    article_text = fetch_response.text.split("</article>")
    
    values_dict = []
    for article in article_text:
        values = bs4.BeautifulSoup(article, "xml")
        
        text_bs = values.find('body')
        abstract_bs = values.find('abstract')
        title_bs = values.find('article-title')
        pmc_id_bs = values.find('article-id', attrs={"pub-id-type":"pmc"})

        text = text_bs.get_text() if text_bs is not None else ""
        abstract = abstract_bs.get_text() if abstract_bs is not None else ""
        title = title_bs.get_text() if title_bs is not None else ""
        pmc_id = pmc_id_bs.get_text() if pmc_id_bs is not None else ""

        values_dict.append({"pmc_id":pmc_id, "title" : title, "text":text, "abstract":abstract})

    return values_dict



