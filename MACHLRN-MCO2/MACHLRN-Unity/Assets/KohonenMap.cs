using System.Collections.Generic;
using UnityEngine;

public class KohonenMap : MonoBehaviour
{
    #region Singleton
    public static KohonenMap Instance;

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

    public List<Material> materialsList = new();
    public GameObject prefab = null;
    public Transform startPos = null;

    public int numRows = 16;
    public int numCols = 16;

    void Start()
    {
        List<List<string>> nodesList = CSVHandler.Instance.nodesList;

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                Vector3 newPos = startPos.position;
                newPos.x += j;
                newPos.y -= i;
                GameObject instance = GameObject.Instantiate(prefab, newPos, Quaternion.identity, startPos);

                int value = int.Parse(nodesList[i][j]);
                Renderer r = instance.GetComponent<Renderer>();
                r.material = materialsList[value - 1];
            }
        }
    }
}
