#!/bin/bash
# Standard internet proxy for IIT Delhi HPC login/interactive nodes.
# Run on the HPC after SSH login: ./proxy.sh &
# Then export:
#   export http_proxy="http://proxy62.iitd.ac.in:3128"
#   export https_proxy="http://proxy62.iitd.ac.in:3128"

proxy_add="https://proxy62.iitd.ernet.in/cgi-bin/proxy.cgi"

username="{kerberos_id}";
password="";   # fill in locally, never commit real password

unsetProxy()
{
        unset http_proxy
        unset https_proxy
        unset ftp_proxy
}

getsessionid()
{
        session_id=`curl $proxy_add -s -k --no-progress-bar --connect-timeout 15| grep -m 1 sessionid[\"=[:alpha:]\ ]*[[:digit:]]* | grep -oh "\"[[:digit:]][[:alnum:]]*\"" | sed 's|"||' | sed 's|"||' `
}

logout()
{
        curl -d "sessionid=$session_id&action=logout" $proxy_add -k -s >/dev/null
        if [ ${#ifAlreadyLoggedIn} = 0 ]; then
                        echo -e "Enjoy!"
        else
                        echo -e "proxy logout!!" ;
        fi

        exit 0
}

login()
{
        logintext=`curl -d "sessionid=$session_id&action=Validate&userid=$username&pass=$password" $proxy_add -k -s`
        ifAlreadyLoggedIn=`echo $logintext | grep -v "already logged in"`
        if [ ${#ifAlreadyLoggedIn} = 0 ]; then
                        echo "Already logged in"
                        exit 0
        fi
}

retries=1000
islogout=1;

mainloop()
{
        unsetProxy
        getsessionid
        login
        iflogin=`echo $logintext | grep "logged in successfully"`
        if [ ${#iflogin} = 0 ] ;then
                echo "error in login"
                sleep 10
                mainloop
        else
                echo "proxy login connected"
                while true; do
                        sleep 120
                        refreshtext=`curl -d "sessionid=$session_id&action=Refresh" $proxy_add -k -s`
                        ifrefresh=`echo $refreshtext | grep "logged in successfully"`
                        if [ ${#ifrefresh} = 0 ] ;
                                then
                                curl -d "sessionid=$session_id&action=logout" $proxy_add -k -s >/dev/null
                                mainloop
                        fi
                done
        fi
}

trap logout EXIT
mainloop
