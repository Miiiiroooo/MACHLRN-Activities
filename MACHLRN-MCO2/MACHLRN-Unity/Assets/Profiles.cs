using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Profiles : MonoBehaviour
{
    public GameObject UI;
    public List<GameObject> legendColors;

    public Transform content;
    public GameObject profilePrefab;


    void Start()
    {
        InitLegends();
        InitProfiles();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            UI.SetActive(!UI.activeSelf);
        }
    }

    void InitLegends()
    {
        List<Material> materialsList = KohonenMap.Instance.materialsList;

        for (int i = 0; i < legendColors.Count; i++)
        {
            Image img = legendColors[i].GetComponent<Image>();
            img.color = materialsList[i].color;
        }
    }

    void InitProfiles()
    {
        List<List<string>> profilesList = CSVHandler.Instance.profilesList;
        int numProfiles = profilesList.Count / 6;

        for (int i = 0; i < numProfiles; i++)
        {
            GameObject instance = GameObject.Instantiate(profilePrefab, content);
            Profile p = instance.GetComponent<Profile>();

            p.textsList[0].text = profilesList[i * 6][0];

            for (int j = 1; j < 6; j++)
            {
                int index = i * 6 + j;
                string str = profilesList[index][1].Replace("%", "");
                float value1 = Mathf.Round(Convert.ToSingle(str) * 100f) / 100f;
                str = profilesList[index][3].Replace("%", "");
                float value2 = Mathf.Round(Convert.ToSingle(str) * 100f) / 100f;

                string msg = profilesList[index][0] + ": " + value1 + "%   " + profilesList[index][2] + ": " + value2 + "%";
                p.textsList[j].text = msg;
            }
        }
    }
}
