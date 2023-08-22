using System.Collections.Generic;
using UnityEngine;

public class CSVHandler : MonoBehaviour
{
    #region Singleton
    public static CSVHandler Instance;

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(this.gameObject);
            return;
        }
        else
        {
            Instance = this;
            Initialize();
        }
    }

    private void OnDestroy()
    {
        if (Instance != null && Instance == this)
        {
            Destroy(this.gameObject);
        }
    }
    #endregion 

    public List<List<string>> nodesList = new();
    public List<List<string>> profilesList = new();

    void Initialize()
    {
        TextAsset dataset = Resources.Load<TextAsset>("nodes");
        if (!dataset)
            return;

        string[] dataLines = dataset.text.Split('\n');  // list of all lines
        for (int i = 0; i < dataLines.Length; i++)
        {
            string[] data = dataLines[i].Split(',');    // list of all comma-separated values
            List<string> newList = new();

            for (int j = 0; j < data.Length; j++)
            {
                newList.Add(data[j]);
            }

            nodesList.Add(newList);
        }

        dataset = Resources.Load<TextAsset>("profiles");
        if (!dataset)
            return;

        dataLines = dataset.text.Split('\n');           // list of all lines
        for (int i = 0; i < dataLines.Length; i++)
        {
            string[] data = dataLines[i].Split(',');    // list of all comma-separated values
            List<string> newList = new();

            for (int j = 0; j < data.Length; j++)
            {
                newList.Add(data[j]);
            }

            profilesList.Add(newList);
        }
    }
}
