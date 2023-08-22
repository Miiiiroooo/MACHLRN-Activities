using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class temp : MonoBehaviour
{
    // Update is called once per frame
    void Update()
    {
        Debug.Log("true:" + transform.forward);
    }

    private void OnDrawGizmos()
    {
        Vector3 dir = transform.forward + transform.position;
        Gizmos.DrawLine(transform.position, dir);

        //Vector3 temp = new Vector3(0.7071f, -0.5f, 0.5f) + transform.position;
        //Gizmos.DrawLine(transform.position, temp);
    }
}
